"""A published checkpoint served in place: its files, its experts' layout, and the rest of the model (Phase 4A).

Decision 0007. What a model adapter needs to stream a checkpoint's experts without converting
or copying it:

  checkpoint_sources     the safetensors files of a pinned revision (its index, or its single
                         file), each with its local path and the sha256 its publisher declares
  expert_sources         for every expert-sliced parameter, the checkpoint tensors it is made of,
                         as name templates with "{expert}". Derived from the model's transformers
                         conversion mapping, i.e. the loader's own declaration. Only stacking one
                         tensor per expert (MergeModulelist(dim=0)), optionally followed by
                         concatenating them along each expert's first dimension (Concatenate(dim=1)),
                         is understood; any other operation (transposes, interleaving,
                         dequantization) is refused rather than guessed. A parameter no converter
                         targets must be a checkpoint tensor of its own name (already stacked).
  expert_index_segments  one segment per expert-sliced parameter, one row per expert. For split
                         tensors, a composed segment: the row is the byte ranges of the expert's
                         tensors in order (concatenating row-major tensors along their first
                         dimension concatenates their bytes). Built from safetensors headers
                         alone, with every shape and dtype checked against the model.
  write_expert_index     an awpmi pack (v2) that holds only that index: nothing is copied, nothing
                         proportional to the model is read
  load_model_without_experts  the model on the device with every weight except the experts,
                         read from the checkpoint with direct reads (so the process never maps
                         the checkpoint through the OS page cache, decision 0006); the experts'
                         parameters stay on `meta`, where any use of them fails
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from awpmi.models.moe import PACK_KIND, ExpertModule, find_expert_modules
from awpmi.storage.layout import ComposedSegment, Segment, safetensors_segments, typed_rows
from awpmi.storage.pack import Pack, PackWriter, SourceFile
from awpmi.storage.store import FileBackedPageStore

EXPERT = "{expert}"


@dataclass(frozen=True)
class CheckpointFile:
    key: str  # the file key segments refer to (the file's name in the repository)
    source: SourceFile
    path: Path
    sha256: str | None  # declared by the publisher (Hugging Face LFS), None if unknown


def checkpoint_sources(repository: str, revision: str, declared_sha256: bool = True) -> dict[str, CheckpointFile]:
    """The safetensors files of a pinned checkpoint revision, found in the local Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    try:
        index = Path(hf_hub_download(repository, "model.safetensors.index.json", revision=revision, local_files_only=True))
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    except Exception:  # no index: a single file
        names = ["model.safetensors"]
    files = {}
    for name in names:
        source = SourceFile(repository, revision, name)
        files[name] = CheckpointFile(name, source, source.resolve(), source.published_sha256() if declared_sha256 else None)
    return files


def checkpoint_tensors(files: Mapping[str, Path]) -> dict[str, Segment]:
    """Every tensor of the checkpoint files (name → segment in its file)."""
    tensors: dict[str, Segment] = {}
    for key, path in files.items():
        for name, segment in safetensors_segments(path, key).items():
            if name in tensors:
                raise ValueError(f"{name} is in two checkpoint files")
            tensors[name] = segment
    return tensors


def checkpoint_renamings(model: nn.Module) -> list[tuple[str, str]]:
    """The literal key renamings of the model's transformers mapping, as (checkpoint text, model text) pairs."""
    from transformers.conversion_mapping import extract_weight_conversions_for_model
    from transformers.core_model_loading import WeightRenaming

    pairs = []
    for transform in extract_weight_conversions_for_model(model) or []:
        if not isinstance(transform, WeightRenaming):
            continue
        for source, target in zip(transform.source_patterns, transform.target_patterns):
            if not _literal(source) or not _literal(target):
                continue  # regex renamings (prefixes, groups) are not reversed: a name must then match as it is
            pairs.append((source, target))
    return pairs


def resolve_tensor_name(name: str, tensors: Mapping[str, object], renamings: list[tuple[str, str]]) -> str | None:
    """The checkpoint name of model tensor `name`: itself, or the one reversed renaming that exists."""
    if name in tensors:
        return name
    found = {name.replace(model_text, checkpoint_text) for checkpoint_text, model_text in renamings if model_text in name}
    found = {candidate for candidate in found if candidate in tensors}
    if len(found) > 1:
        raise ValueError(f"{name}: several checkpoint tensors match ({sorted(found)})")
    return found.pop() if found else None


def expert_sources(model: nn.Module, modules: list[ExpertModule] | None = None) -> dict[str, dict[str, tuple[str, ...]]]:
    """Experts module → parameter → model-side tensor templates (see the module docstring).

    The templates use the model's own names; `expert_index_segments` finds them in the
    checkpoint, reversing the mapping's literal renamings where a checkpoint uses older names.
    """
    from transformers.conversion_mapping import extract_weight_conversions_for_model
    from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter

    modules = find_expert_modules(model) if modules is None else modules
    converters = [c for c in (extract_weight_conversions_for_model(model) or []) if isinstance(c, WeightConverter)]
    sources: dict[str, dict[str, tuple[str, ...]]] = {}
    for entry in modules:
        sources[entry.name] = {}
        for parameter in entry.parameters:
            full = f"{entry.name}.{parameter}"
            matching = [c for c in converters if any(_is_suffix(full, target) for target in c.target_patterns)]
            if not matching:
                sources[entry.name][parameter] = (full,)  # stored stacked, under its own name
                continue
            if len(matching) > 1 or len(matching[0].target_patterns) != 1:
                raise ValueError(f"{full}: several conversions target it")
            converter = matching[0]
            operations = converter.operations
            kinds = [(type(op), getattr(op, "dim", None)) for op in operations]
            if kinds not in ([(MergeModulelist, 0)], [(MergeModulelist, 0), (Concatenate, 1)]):
                raise ValueError(f"{full}: unsupported conversion {operations} (only per-expert stacking and concatenation)")
            target = converter.target_patterns[0]
            head = full[: len(full) - len(target)]
            templates = []
            for pattern in converter.source_patterns:
                if pattern.count("*") != 1 or not _literal(pattern.replace("*", "")):
                    raise ValueError(f"{full}: unsupported source pattern {pattern!r}")
                templates.append(head + pattern.replace("*", EXPERT))
            sources[entry.name][parameter] = tuple(templates)
    return sources


def parameter_segments(model: nn.Module, parameters: list[str], files: Mapping[str, Path]) -> dict[str, Segment]:
    """Each named parameter's checkpoint tensor, as it is, as a segment named after the parameter (headers only).

    For dense weights served from storage (`awpmi.models.streamed`); a checkpoint's older names are resolved by the
    mapping's literal renamings, as for the experts.
    """
    tensors = checkpoint_tensors(files)
    renamings = checkpoint_renamings(model)
    segments = {}
    for name in parameters:
        resolved = resolve_tensor_name(name, tensors, renamings)
        if resolved is None:
            raise KeyError(f"the checkpoint has no tensor {name}")
        found = tensors[resolved]
        segments[name] = Segment(name, found.file, found.offset, found.rows, found.row_bytes, found.dtype, found.row_shape)
    return segments


def checked_expert_sources(
    model: nn.Module,
    layout: Mapping[str, tuple[str, ...]],
    experts_suffix: str,
    modules: list[ExpertModule] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """`expert_sources`, required to equal an adapter's own statement of the layout (templates relative to the layer).

    A model adapter states the layout it expects (`layout`: parameter → checkpoint tensor templates
    after the layer prefix); a change on either side is then an error, not wrong bytes streamed.
    """
    modules = find_expert_modules(model) if modules is None else modules
    derived = expert_sources(model, modules)
    for entry in modules:
        if not entry.name.endswith(experts_suffix):
            raise ValueError(f"unexpected experts module {entry.name}")
        head = entry.name[: -len(experts_suffix)]
        expected = {parameter: tuple(head + t for t in templates) for parameter, templates in layout.items()}
        if derived[entry.name] != expected:
            raise ValueError(f"{entry.name}: transformers maps {derived[entry.name]}, the adapter's layout is {expected}")
    return derived


def neighbours(model: nn.Module, experts_suffix: str, suffix: str, modules: list[ExpertModule] | None = None) -> dict[str, nn.Module]:
    """Experts module name → the module at `suffix` in the same layer (its router, its shared experts, its MoE block)."""
    modules = find_expert_modules(model) if modules is None else modules
    found = {}
    for entry in modules:
        if not entry.name.endswith(experts_suffix):
            raise ValueError(f"unexpected experts module {entry.name}")
        head = entry.name[: -len(experts_suffix)]
        found[entry.name] = model.get_submodule((head + suffix).rstrip("."))
    return found


def _literal(pattern: str) -> bool:
    """A pattern with no regex syntax but '.' (which transformers' patterns use for a literal dot)."""
    return not re.search(r"[\^$*+?()\[\]{}|\\]", pattern)


def _is_suffix(name: str, pattern: str) -> bool:
    if not _literal(pattern):
        return False
    if pattern.startswith("."):
        return name.endswith(pattern)
    return name == pattern or name.endswith("." + pattern)


def expert_index_segments(
    modules: list[ExpertModule],
    sources: Mapping[str, Mapping[str, tuple[str, ...]]],
    tensors: Mapping[str, Segment],
    renamings: list[tuple[str, str]] = (),
) -> dict[str, Segment | ComposedSegment]:
    """Segment name (module.parameter) → its segment in the checkpoint, checked against the model's shapes and dtypes."""
    from awpmi.storage.layout import DTYPE_NAMES

    segments: dict[str, Segment | ComposedSegment] = {}
    for entry in modules:
        for parameter in entry.parameters:
            name = entry.segment(parameter)
            template = getattr(entry.module, parameter)
            dtype = DTYPE_NAMES[template.dtype]
            shape = tuple(template.shape)  # [E, *row]
            templates = sources[entry.name][parameter]
            if len(templates) == 1 and EXPERT not in templates[0]:
                resolved = resolve_tensor_name(templates[0], tensors, list(renamings))
                found = None if resolved is None else tensors[resolved]
                if found is None or found.shape != shape or found.dtype != dtype:
                    raise ValueError(f"{name}: no checkpoint tensor {templates[0]} of shape {shape} and dtype {dtype}")
                segments[name] = Segment(name, found.file, found.offset, found.rows, found.row_bytes, found.dtype, found.row_shape)
                continue
            # The checkpoint's names for these templates, chosen once (on expert 0) and kept for every expert.
            resolved_templates = []
            for pattern in templates:
                first = resolve_tensor_name(pattern.replace(EXPERT, "0"), tensors, list(renamings))
                if first is None:
                    raise ValueError(f"{name}: the checkpoint has no {pattern.replace(EXPERT, '0')}")
                for checkpoint_text, model_text in [("", ""), *renamings]:
                    candidate = pattern.replace(model_text, checkpoint_text) if model_text else pattern
                    if candidate.replace(EXPERT, "0") == first:
                        resolved_templates.append(candidate)
                        break
            spans, part_bytes = [], None
            for expert in range(entry.num_experts):
                parts = []
                for pattern in resolved_templates:
                    found = tensors.get(pattern.replace(EXPERT, str(expert)))
                    if found is None:
                        raise ValueError(f"{name}: the checkpoint has no {pattern.replace(EXPERT, str(expert))}")
                    if found.dtype != dtype or found.row_shape != shape[2:]:
                        raise ValueError(f"{name}: {found.name} is {found.dtype} {found.shape}, not a slice of {dtype} {shape}")
                    parts.append(found)
                if sum(part.rows for part in parts) != shape[1]:
                    raise ValueError(f"{name}: expert {expert}'s tensors have {sum(p.rows for p in parts)} rows, not {shape[1]}")
                sizes = tuple(part.nbytes for part in parts)
                if part_bytes is not None and sizes != part_bytes:
                    raise ValueError(f"{name}: experts have tensors of different sizes")
                part_bytes = sizes
                spans.append(tuple((part.file, part.offset) for part in parts))
            row_bytes = sum(part_bytes)
            segments[name] = ComposedSegment(name, entry.num_experts, row_bytes, dtype, shape[1:], part_bytes, tuple(spans))
    return segments


def write_expert_index(
    model: nn.Module,
    directory: str | Path,
    files: Mapping[str, CheckpointFile],
    packing: dict | None = None,
    metadata: dict | None = None,
    sources: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
) -> Pack:
    """An awpmi pack of `model`'s experts that refers to the checkpoint files in place (headers only).

    `sources` overrides the layout derived from transformers (an adapter's checked layout).
    """
    modules = find_expert_modules(model)
    if not modules:
        raise ValueError("the model has no experts module")
    sources = expert_sources(model, modules) if sources is None else sources
    tensors = checkpoint_tensors({k: f.path for k, f in files.items()})
    segments = expert_index_segments(modules, sources, tensors, checkpoint_renamings(model))
    writer = PackWriter(directory, PACK_KIND)
    used = {file for segment in segments.values() for file in segment.files}
    for key in sorted(used):
        writer.add_source(key, files[key].source, files[key].path, sha256=files[key].sha256)
    for segment in segments.values():
        if isinstance(segment, ComposedSegment):
            writer.add_composed_segment(segment)
        else:
            raise ValueError(f"{segment.name} is stacked in the checkpoint: use write_expert_pack, which verifies its bytes")
    info = {
        "groups": {
            m.name: {"experts": m.num_experts, "segments": {p: m.segment(p) for p in m.parameters}} for m in modules
        },
        "layout": {m.name: {p: list(sources[m.name][p]) for p in m.parameters} for m in modules},
        **(metadata or {}),
    }
    return writer.write(info, packing)


def load_model_without_experts(
    repository: str,
    revision: str,
    dtype: torch.dtype,
    device: torch.device | str,
    files: Mapping[str, Path],
    model_class=None,
) -> tuple[nn.Module, dict]:
    """The model with every weight but the experts on `device`; experts stay on `meta`. Returns (model, report).

    The skeleton is built on `meta` from the config (transformers' default kernels, as
    `from_pretrained` would pick); every other parameter and persistent buffer is the
    checkpoint tensor of the same name, read with direct I/O and cast as `from_pretrained`
    casts it (its dtype plan, e.g. `_keep_in_fp32_modules_strict`, then the declared dtype);
    non-persistent buffers (e.g. the rotary frequencies) are recomputed by the model's own
    initializer, as `from_pretrained` does. A weight the checkpoint does not hold under its own
    name (a conversion other than the experts') is an error, and so is an expert parameter the
    dtype plan would cast: streamed expert bytes are used as stored.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.core_model_loading import build_glob_alternation

    config = AutoConfig.from_pretrained(repository, revision=revision)
    with torch.device("meta"):
        model = (model_class or AutoModelForCausalLM).from_config(config, dtype=dtype)
    model.eval()
    modules = find_expert_modules(model)
    skipped = {f"{m.name}.{p}" for m in modules for p in m.parameters}
    plan = model._get_dtype_plan(dtype)
    policy = build_glob_alternation(list(plan))[:2] if plan else None

    def planned_dtype(name: str, declared: torch.dtype) -> torch.dtype:
        # transformers' rule (core_model_loading): the dtype plan, else the declared dtype if it differs.
        match = policy[0].search(name) if policy else None
        if match is not None:
            return plan[policy[1][match.lastgroup]]
        return declared

    for name in skipped:
        if policy and policy[0].search(name):
            raise ValueError(f"{name}: an expert parameter that transformers would cast cannot be streamed as stored")
    tensors = checkpoint_tensors(files)
    renamings = checkpoint_renamings(model)
    wanted: dict[str, Segment] = {}
    targets = []
    for name, tensor in [*model.named_parameters(), *_persistent_buffers(model)]:
        if name in skipped:
            continue
        resolved = resolve_tensor_name(name, tensors, renamings)
        if resolved is None:
            raise KeyError(f"the checkpoint has no tensor {name}")
        found = tensors[resolved]
        if found.shape != tuple(tensor.shape):
            raise ValueError(f"{name}: checkpoint {found.shape} differs from the model's {tuple(tensor.shape)}")
        wanted[name] = found
        targets.append((name, tensor))
    store = FileBackedPageStore(dict(files), wanted, direct=True)
    loaded = 0
    try:
        for name, tensor in targets:
            segment = wanted[name]
            value = typed_rows(store.read_rows(name), segment).reshape(tensor.shape)
            value = value.to(device=device, dtype=planned_dtype(name, tensor.dtype))
            value._is_hf_initialized = True  # transformers' initializers leave flagged tensors alone
            _assign(model, name, value, isinstance(tensor, nn.Parameter))
            loaded += segment.nbytes
    finally:
        store.close()
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    for owner in sorted({name.rsplit(".", 1)[0] for name, _ in model.named_buffers() if _non_persistent(model, name)}):
        module = model.get_submodule(owner)
        for buffer in [b for b in module._non_persistent_buffers_set if module._buffers.get(b) is not None]:
            module._buffers[buffer] = torch.empty_like(module._buffers[buffer], device=device)
        model._init_weights(module)
    leftovers = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta and n not in skipped]
    if leftovers:
        raise RuntimeError(f"weights left on meta: {leftovers[:5]}")
    report = {"loaded_bytes": loaded, "tensors": len(targets), "expert_parameters": sorted(skipped)}
    return model, report


def _persistent_buffers(model: nn.Module):
    for name, buffer in model.named_buffers():
        if buffer is not None and not _non_persistent(model, name):
            yield name, buffer


def _non_persistent(model: nn.Module, name: str) -> bool:
    owner, _, buffer = name.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    return buffer in module._non_persistent_buffers_set


def _assign(model: nn.Module, name: str, value: torch.Tensor, parameter: bool) -> None:
    owner, _, attribute = name.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    if parameter:
        parameter_value = nn.Parameter(value, requires_grad=False)
        parameter_value._is_hf_initialized = True
        module._parameters[attribute] = parameter_value
    else:
        module._buffers[attribute] = value
