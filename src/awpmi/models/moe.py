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
class StreamedExperts:
    """Serve a model's experts from an `ExpertStore`; see the module docstring."""

    model: nn.Module
    experts: ExpertStore
    buffer_sets: int = 1
    poison: bool = False
    on_route: Callable[[RoutingRecord], None] | None = None
    keep_resident: bool = False
    all_experts: bool = False  # serve every expert of every layer (dense layer streaming, the baseline)
    _modules: list[ExpertModule] = field(default_factory=list, init=False)
    _handles: list = field(default_factory=list, init=False)
    _originals: dict = field(default_factory=dict, init=False)
    _buffers: dict = field(default_factory=dict, init=False)

    def install(self) -> StreamedExperts:
        if self._handles:
            raise RuntimeError("already installed")
        self._modules = find_expert_modules(self.model)
        missing = {m.name for m in self._modules} - set(self.experts.groups)
        if missing:
            raise KeyError(f"the expert store has no group for {sorted(missing)}")
        for index, entry in enumerate(self._modules):
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
        return self

    def _hook(self, entry: ExpertModule, buffers: dict[str, torch.Tensor]):
        def hook(module, args, kwargs):
            top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
            experts = torch.unique(top_k_index.detach())
            experts = experts[(experts >= 0) & (experts < entry.num_experts)]  # drop sentinels
            routed = experts.tolist()
            if self.all_experts:
                experts = torch.arange(entry.num_experts, device=experts.device)
            if self.poison:
                for buffer in buffers.values():
                    buffer.fill_(float("nan") if buffer.is_floating_point() else -1)
            self.experts.fill(entry.name, experts, buffers)
            if self.on_route is not None:
                hidden = args[0] if args else kwargs["hidden_states"]
                self.on_route(RoutingRecord(entry.name, routed, hidden.shape[0]))
            return None

        return hook

    def remove(self) -> None:
        """Remove the hooks; with `keep_resident`, restore the original parameters."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self.keep_resident:
            for entry in self._modules:
                for parameter in entry.parameters:
                    entry.module._parameters[parameter] = self._originals[(entry.name, parameter)]
            self._originals.clear()
        self._buffers.clear()

    @property
    def modules(self) -> list[ExpertModule]:
        return list(self._modules)

    @property
    def buffer_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self._buffers.values())
