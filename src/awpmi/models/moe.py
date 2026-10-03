"""Mixture-of-experts adapter for Hugging Face transformers models (Phase 3, decision 0006).

transformers 5 gives every MoE model one experts convention (`use_experts_implementation`): an
experts module has an integer `num_experts`, parameters whose first dimension is the expert
index (e.g. gate_up_proj [E, 2I, H], down_proj [E, H, I], biases [E, out]), and is called as
experts(hidden_states, top_k_index, top_k_weights) after the router has chosen. This adapter
relies on that convention only. No model name, tensor name or expert count is written here:

  * `find_expert_modules` finds every module with `num_experts` and at least one parameter of
    three or more dimensions whose first dimension is `num_experts` (a router's [E, H] weight
    is not one); every parameter of that module with first dimension `num_experts` is
    expert-sliced.
  * `write_expert_pack` stores each expert-sliced parameter as one segment with a row per
    expert. A checkpoint tensor with exactly the parameter's bytes (the loader only renamed
    it) is referred to instead of copied, so experts are read from the published file.
  * `StreamedExperts` replaces each expert-sliced parameter by a device buffer of the same
    shape and registers a forward pre-hook. The hook reads the router's choice (top_k_index),
    asks the `ExpertStore` for those experts of that layer, and writes them into their rows
    of the buffer; then the module's own forward runs, unchanged, on the same bytes as the
    resident model. Rows of experts that were not routed hold stale data of another layer:
    the experts implementations read only routed experts, which the poison mode checks by
    filling every unrouted row with NaN first.

Buffers are shared by layers in round robin (`buffer_sets`). Every write into them is issued
on the compute stream after the previous layer's kernels, so one set is already correct;
more sets leave room for prefetching the next layer while one computes.

Compact mode (Phase 4A, decision 0007) drops the full-shape buffers, which for a
DeepSeek-class layer would hold E × expert bytes. Between calls the expert-sliced parameters
are None: the full tensors exist nowhere, and any use of them raises. (A `meta` placeholder
would not do: a CUDA grouped GEMM given a meta weight returned garbage instead of failing.)
For each call the pre-hook takes the routed experts R (ascending, each once however many tokens chose it),
assembles them into fresh buffers [|R|, ...] (row i = expert R[i]), sets the parameters to
those buffers and `num_experts` to |R|, and replaces top_k_index by the slot of each expert;
a forward hook restores the placeholders. The slots keep the experts' order, which every
experts implementation needs to stay bit for bit the reference's: eager accumulates in
expert order, grouped_mm sorts by id and then sums the top-k in their own order. Device
memory then follows the routed (and cached) experts, not the expert count. Poison mode adds
spare slots and fills every slot with NaN before assembly: the spares must never be read,
and assembly must write every routed byte.

`FullLayerOffload` is the reference execution for a model whose experts do not fit on the
device: each experts module's own parameters stay in host memory (as the transformers loader
made them), are copied whole to the device just before the module runs, and the module's
forward runs unchanged on them. It is the fully materialized model, one layer at a time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from awpmi.materialization.weights import ExpertGroup, ExpertStore
from awpmi.storage.layout import safetensors_segments
from awpmi.storage.pack import Pack, PackWriter, SourceFile, tensor_bytes_sha256
from awpmi.storage.store import FileBackedPageStore
from awpmi.tracing import tensor_digest

PACK_KIND = "moe-experts"
POISON_SPARE_SLOTS = 2


@dataclass(frozen=True)
class ExpertModule:
    name: str  # module path in the model
    module: nn.Module
    num_experts: int
    parameters: tuple[str, ...]  # expert-sliced parameters, first dimension = num_experts

    def segment(self, parameter: str) -> str:
        return f"{self.name}.{parameter}"


def find_expert_modules(model: nn.Module) -> list[ExpertModule]:
    """Every experts module of the model, in module order (see the module docstring for the rule)."""
    found = []
    for name, module in model.named_modules():
        count = getattr(module, "num_experts", None)
        if not isinstance(count, int) or count <= 0:
            continue
        parameters = {key: value for key, value in module.named_parameters(recurse=False) if value.dim() and value.shape[0] == count}
        if any(value.dim() >= 3 for value in parameters.values()):
            found.append(ExpertModule(name, module, count, tuple(parameters)))
    return found


def move_except_experts(model: nn.Module, device: torch.device | str) -> int:
    """Move every parameter and buffer to `device` except the expert-sliced parameters; returns the bytes moved."""
    experts = {(id(m.module), p) for m in find_expert_modules(model) for p in m.parameters}
    moved = 0
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None or (id(module), name) in experts or parameter.device == torch.device(device):
                continue
            module._parameters[name] = nn.Parameter(parameter.detach().to(device), requires_grad=False)
            moved += parameter.numel() * parameter.element_size()
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
                moved += buffer.numel() * buffer.element_size()
    return moved


def expert_groups(modules: list[ExpertModule]) -> dict[str, ExpertGroup]:
    return {
        m.name: ExpertGroup(m.name, m.num_experts, {parameter: m.segment(parameter) for parameter in m.parameters})
        for m in modules
    }


def _source_tensors_by_digest(files: Mapping[str, Path], wanted: Mapping[str, tuple[torch.Size, torch.dtype, int]]) -> dict[str, tuple[str, str]]:
    """sha256 → (source key, tensor name) for every checkpoint tensor whose size could match a wanted one."""
    sizes = {nbytes for _, _, nbytes in wanted.values()}
    found = {}
    for key, path in files.items():
        segments = safetensors_segments(path, key)
        candidates = {name: s for name, s in segments.items() if s.nbytes in sizes}
        if not candidates:
            continue
        reader = FileBackedPageStore({key: path}, candidates, direct=True)
        try:
            for name, segment in candidates.items():
                digest = hashlib.sha256()
                step = max(1, (64 << 20) // segment.row_bytes)
                for first in range(0, segment.rows, step):
                    rows = torch.arange(first, min(first + step, segment.rows))
                    digest.update(reader.read_rows(name, rows).numpy().tobytes())
                found.setdefault(digest.hexdigest(), (key, name))
        finally:
            reader.close()
    return found


def write_expert_pack(
    model: nn.Module,
    directory: str | Path,
    sources: Mapping[str, tuple[SourceFile, Path | None]] | None = None,
    packing: dict | None = None,
) -> Pack:
    """Pack every expert-sliced parameter of `model`; refer to checkpoint tensors that hold the same bytes.

    `sources` maps a key to a checkpoint file (and optionally its local path); without
    sources, or for a parameter no checkpoint tensor matches, the parameter is copied.
    """
    modules = find_expert_modules(model)
    if not modules:
        raise ValueError("the model has no experts module")
    writer = PackWriter(directory, PACK_KIND)
    paths = {}
    for key, (source, path) in (sources or {}).items():
        writer.add_source(key, source, path)
        paths[key] = Path(path) if path is not None else source.resolve()
    parameters = {m.segment(p): getattr(m.module, p).detach() for m in modules for p in m.parameters}
    digests = {name: tensor_bytes_sha256(tensor) for name, tensor in parameters.items()}
    wanted = {name: (tensor.shape, tensor.dtype, tensor.numel() * tensor.element_size()) for name, tensor in parameters.items()}
    matches = _source_tensors_by_digest(paths, wanted) if paths else {}
    referenced = {}
    for name, tensor in parameters.items():
        match = matches.get(digests[name])
        if match is not None:
            writer.add_source_segment(name, match[0], match[1], expected=tensor)
            referenced[name] = f"{match[0]}:{match[1]}"
        else:
            writer.add_tensor(name, tensor)
    metadata = {
        "groups": {
            m.name: {"experts": m.num_experts, "segments": {p: m.segment(p) for p in m.parameters}} for m in modules
        },
        "referenced": referenced,
        "experts_sha256": tensor_digest(*parameters.values()),
    }
    return writer.write(metadata, packing)


def record_routing(model: nn.Module, on_route: Callable[[RoutingRecord], None]) -> list:
    """Observe the routed experts of every experts module (resident model); returns the hook handles."""
    handles = []
    for entry in find_expert_modules(model):

        def hook(module, args, kwargs, entry=entry):
            top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
            experts = torch.unique(top_k_index.detach())
            experts = experts[(experts >= 0) & (experts < entry.num_experts)]
            hidden = args[0] if args else kwargs["hidden_states"]
            on_route(RoutingRecord(entry.name, experts.tolist(), hidden.shape[0]))

        handles.append(entry.module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def groups_from_pack(pack: Pack) -> dict[str, ExpertGroup]:
    if pack.kind != PACK_KIND:
        raise ValueError(f"{pack.directory} is a {pack.kind!r} pack")
    return {
        key: ExpertGroup(key, entry["experts"], dict(entry["segments"])) for key, entry in pack.metadata["groups"].items()
    }


@dataclass
class RoutingRecord:
    """One experts call: which module, which experts were routed (ascending), how many tokens."""

    module: str
    experts: list[int]
    tokens: int


@dataclass
class ExpertCall:
    """One experts call as it ran, seen before its weights are released (for audits and records).

    `call_index` is what the module received: slot ids in compact mode, expert ids otherwise.
    `served` lists the experts present for the call, in slot order (every expert for a
    full-shape or offloaded module).
    """

    module: str
    experts_module: nn.Module
    hidden_states: torch.Tensor
    top_k_index: torch.Tensor
    top_k_weights: torch.Tensor
    call_index: torch.Tensor
    served: list[int]
    output: torch.Tensor

    def per_assignment_outputs(self) -> torch.Tensor:
        """Every (token, k) assignment's expert output before routing weights, [tokens·k, hidden].

        The module runs again on the same weights (no hooks), with each assignment as a token
        routed to its expert alone with weight 1: the experts group the same rows per expert
        as in the call, and multiplying by 1 and summing one term are exact.
        """
        top_k = self.top_k_index.shape[-1]
        hidden = self.hidden_states.repeat_interleave(top_k, dim=0)
        index = self.call_index.reshape(-1, 1)
        ones = torch.ones(index.shape[0], 1, dtype=self.top_k_weights.dtype, device=index.device)
        return self.experts_module.forward(hidden, index, ones)


_CALL_ARGUMENTS = ("hidden_states", "top_k_index", "top_k_weights")


def _call_argument(args: tuple, kwargs: dict, position: int):
    return args[position] if len(args) > position else kwargs[_CALL_ARGUMENTS[position]]


def _replace_argument(args: tuple, kwargs: dict, position: int, value) -> tuple[tuple, dict]:
    if len(args) > position:
        return (*args[:position], value, *args[position + 1 :]), kwargs
    return args, {**kwargs, _CALL_ARGUMENTS[position]: value}


def routed_experts(top_k_index: torch.Tensor, num_experts: int) -> torch.Tensor:
    """The experts a call routes to, ascending and unique (sentinels outside [0, E) dropped)."""
    experts = torch.unique(top_k_index.detach())
    return experts[(experts >= 0) & (experts < num_experts)]


@dataclass
class StreamedExperts:
    """Serve a model's experts from an `ExpertStore`; see the module docstring."""

    model: nn.Module
    experts: ExpertStore
    buffer_sets: int = 1
    poison: bool = False
    on_route: Callable[[RoutingRecord], None] | None = None
    keep_resident: bool = False
    all_experts: bool = False  # serve every expert of every layer (dense layer streaming, the baseline)
    compact: bool = False  # buffers for the routed experts only (decision 0007)
    on_call: Callable[[ExpertCall], None] | None = None
    _modules: list[ExpertModule] = field(default_factory=list, init=False)
    _handles: list = field(default_factory=list, init=False)
    _originals: dict = field(default_factory=dict, init=False)
    _buffers: dict = field(default_factory=dict, init=False)
    _placeholders: dict = field(default_factory=dict, init=False)
    _active: dict = field(default_factory=dict, init=False)
    compact_bytes: int = field(default=0, init=False)  # bytes of the compact buffers of the call in flight
    peak_compact_bytes: int = field(default=0, init=False)

    def install(self) -> StreamedExperts:
        if self._handles:
            raise RuntimeError("already installed")
        self._modules = find_expert_modules(self.model)
        missing = {m.name for m in self._modules} - set(self.experts.groups)
        if missing:
            raise KeyError(f"the expert store has no group for {sorted(missing)}")
        for index, entry in enumerate(self._modules):
            if self.compact:
                self._install_compact(entry)
                continue
            buffers = {}
            for parameter in entry.parameters:
                original = getattr(entry.module, parameter)
                key = (index % self.buffer_sets, tuple(original.shape), original.dtype, str(original.device), parameter)
                if key not in self._buffers:
                    self._buffers[key] = torch.empty_like(original, requires_grad=False)
                buffers[parameter] = self._buffers[key]
                if self.keep_resident:
                    self._originals[(entry.name, parameter)] = original
                entry.module._parameters[parameter] = nn.Parameter(buffers[parameter], requires_grad=False)
            self._handles.append(entry.module.register_forward_pre_hook(self._hook(entry, buffers), with_kwargs=True))
            if self.on_call is not None:
                self._handles.append(entry.module.register_forward_hook(self._observe_full(entry), with_kwargs=True))
        return self

    def _hook(self, entry: ExpertModule, buffers: dict[str, torch.Tensor]):
        def hook(module, args, kwargs):
            top_k_index = _call_argument(args, kwargs, 1)
            experts = routed_experts(top_k_index, entry.num_experts)
            routed = experts.tolist()
            if self.all_experts:
                experts = torch.arange(entry.num_experts, device=experts.device)
            if self.poison:
                for buffer in buffers.values():
                    buffer.fill_(float("nan") if buffer.is_floating_point() else -1)
            self.experts.fill(entry.name, experts, buffers)
            if self.on_route is not None:
                self.on_route(RoutingRecord(entry.name, routed, _call_argument(args, kwargs, 0).shape[0]))
            return None

        return hook

    def _observe_full(self, entry: ExpertModule):
        def hook(module, args, kwargs, output):
            index = _call_argument(args, kwargs, 1)
            self.on_call(
                ExpertCall(
                    entry.name, module, _call_argument(args, kwargs, 0), index, _call_argument(args, kwargs, 2), index,
                    list(range(entry.num_experts)), output,
                )
            )

        return hook

    # Compact mode

    def _install_compact(self, entry: ExpertModule) -> None:
        shapes = {}
        for parameter in entry.parameters:
            original = getattr(entry.module, parameter)
            if self.keep_resident:
                self._originals[(entry.name, parameter)] = original
            shapes[parameter] = (tuple(original.shape), original.dtype)
            entry.module._parameters[parameter] = None
        self._placeholders[entry.name] = shapes
        self._handles.append(entry.module.register_forward_pre_hook(self._compact_hook(entry), with_kwargs=True))
        self._handles.append(entry.module.register_forward_hook(self._compact_release(entry), with_kwargs=True, always_call=True))

    def _compact_hook(self, entry: ExpertModule):
        count = entry.num_experts

        def hook(module, args, kwargs):
            hidden = _call_argument(args, kwargs, 0)
            top_k_index = _call_argument(args, kwargs, 1)
            routed = routed_experts(top_k_index, count)
            served = torch.arange(count, device=routed.device) if self.all_experts else routed
            slots = served.numel() + (POISON_SPARE_SLOTS if self.poison else 0)
            buffers = {}
            for parameter, (shape, dtype) in self._placeholders[entry.name].items():
                buffers[parameter] = torch.empty((slots, *shape[1:]), dtype=dtype, device=hidden.device)
                if self.poison:
                    buffers[parameter].fill_(float("nan") if dtype.is_floating_point else -1)
            self.experts.assemble(entry.name, served.cpu(), buffers)
            nbytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers.values())
            self.compact_bytes += nbytes
            self.peak_compact_bytes = max(self.peak_compact_bytes, self.compact_bytes)
            # Slot of each expert; anything outside [0, E) (a sentinel) maps to the new expert count.
            slot_of = torch.full((count,), slots, dtype=top_k_index.dtype, device=top_k_index.device)
            slot_of[served.to(top_k_index.device)] = torch.arange(served.numel(), dtype=top_k_index.dtype, device=top_k_index.device)
            valid = (top_k_index >= 0) & (top_k_index < count)
            call_index = torch.where(valid, slot_of[top_k_index.clamp(0, count - 1)], torch.full_like(top_k_index, slots))
            for parameter, buffer in buffers.items():
                module._parameters[parameter] = nn.Parameter(buffer, requires_grad=False)
            module.num_experts = slots
            self._active[entry.name] = (top_k_index, call_index, served.tolist(), nbytes)
            if self.on_route is not None:
                self.on_route(RoutingRecord(entry.name, routed.tolist(), hidden.shape[0]))
            return _replace_argument(args, kwargs, 1, call_index)

        return hook

    def _compact_release(self, entry: ExpertModule):
        def hook(module, args, kwargs, output):
            active = self._active.pop(entry.name, None)
            try:
                if active is not None and self.on_call is not None:
                    top_k_index, call_index, served, _ = active
                    self.on_call(
                        ExpertCall(
                            entry.name, module, _call_argument(args, kwargs, 0), top_k_index, _call_argument(args, kwargs, 2),
                            call_index, served, output,
                        )
                    )
            finally:
                for parameter in self._placeholders[entry.name]:
                    module._parameters[parameter] = None
                module.num_experts = entry.num_experts
                if active is not None:
                    self.compact_bytes -= active[3]

        return hook

    def reset_peak(self) -> None:
        self.peak_compact_bytes = self.compact_bytes

    def remove(self) -> None:
        """Remove the hooks; with `keep_resident`, restore the original parameters.

        Otherwise compact mode leaves `meta` parameters of the experts' shapes, so the model
        still shows its experts modules (to install another server) but holds no expert bytes.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self.keep_resident:
            for entry in self._modules:
                for parameter in entry.parameters:
                    entry.module._parameters[parameter] = self._originals[(entry.name, parameter)]
            self._originals.clear()
        elif self.compact:
            for entry in self._modules:
                for parameter, (shape, dtype) in self._placeholders[entry.name].items():
                    entry.module._parameters[parameter] = nn.Parameter(torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False)
        self._buffers.clear()

    @property
    def modules(self) -> list[ExpertModule]:
        return list(self._modules)

    @property
    def buffer_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self._buffers.values())


@dataclass
class FullLayerOffload:
    """The fully materialized reference, one experts layer at a time; see the module docstring.

    Experts modules listed in `resident` are moved to the device once and stay there. The
    others keep their parameters in host memory (pinned with `pin`); before each call they are
    copied whole into device buffers shared by the offloaded layers (in stream order, after
    the previous layer's kernels), and the module's own forward runs on them.
    """

    model: nn.Module
    device: torch.device | str
    resident: tuple[str, ...] = ()
    pin: bool = False
    on_call: Callable[[ExpertCall], None] | None = None
    _modules: list[ExpertModule] = field(default_factory=list, init=False)
    _handles: list = field(default_factory=list, init=False)
    _host: dict = field(default_factory=dict, init=False)
    _buffers: dict = field(default_factory=dict, init=False)
    copied_bytes: int = field(default=0, init=False)

    def install(self) -> FullLayerOffload:
        self._modules = find_expert_modules(self.model)
        unknown = set(self.resident) - {m.name for m in self._modules}
        if unknown:
            raise KeyError(f"no experts modules {sorted(unknown)}")
        for entry in self._modules:
            for parameter in entry.parameters:
                tensor = getattr(entry.module, parameter).detach()
                if entry.name in self.resident:
                    entry.module._parameters[parameter] = nn.Parameter(tensor.to(self.device), requires_grad=False)
                    continue
                tensor = tensor.to("cpu")
                if self.pin and not tensor.is_pinned():
                    tensor = tensor.pin_memory()
                self._host[(entry.name, parameter)] = nn.Parameter(tensor, requires_grad=False)
                entry.module._parameters[parameter] = self._host[(entry.name, parameter)]
            if entry.name not in self.resident:
                self._handles.append(entry.module.register_forward_pre_hook(self._materialize(entry), with_kwargs=True))
            self._handles.append(entry.module.register_forward_hook(self._release(entry), with_kwargs=True, always_call=True))
        return self

    def _materialize(self, entry: ExpertModule):
        def hook(module, args, kwargs):
            for parameter in entry.parameters:
                host = self._host[(entry.name, parameter)]
                key = (parameter, tuple(host.shape), host.dtype)
                if key not in self._buffers:
                    self._buffers[key] = torch.empty(host.shape, dtype=host.dtype, device=self.device)
                buffer = self._buffers[key]
                buffer.copy_(host, non_blocking=self.pin)
                self.copied_bytes += buffer.numel() * buffer.element_size()
                module._parameters[parameter] = nn.Parameter(buffer, requires_grad=False)
            return None

        return hook

    def _release(self, entry: ExpertModule):
        def hook(module, args, kwargs, output):
            try:
                if self.on_call is not None:
                    index = _call_argument(args, kwargs, 1)
                    self.on_call(
                        ExpertCall(
                            entry.name, module, _call_argument(args, kwargs, 0), index, _call_argument(args, kwargs, 2), index,
                            list(range(entry.num_experts)), output,
                        )
                    )
            finally:
                if entry.name not in self.resident:
                    for parameter in entry.parameters:
                        module._parameters[parameter] = self._host[(entry.name, parameter)]

        return hook

    def remove(self) -> None:
        """Remove the hooks; the modules keep their host parameters (resident ones stay on the device)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._buffers.clear()
        self._host.clear()  # the modules hold their host parameters; no second reference keeps a copy alive

    @property
    def device_buffer_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self._buffers.values())
