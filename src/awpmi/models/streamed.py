"""Dense parameters served from storage at every call of their module (Phase 4B, decision 0008).

`StreamedParameters` takes chosen parameters of a model (a MoE layer's shared experts, say) and serves them from a
`WeightStore` instead of keeping them resident. Between calls of their module the parameters are None, so any other use
raises; a pre-hook materializes each one whole (its segment is the checkpoint tensor as it is: a weight [out, in] is out
rows), sets it, and a hook releases them once the module has run. Nothing about the model is assumed beyond parameter
names and the segments given for them; shapes and dtypes are checked against the segments.

It is the measured alternative to residency: every call pays the parameters' bytes from the cache or the drive, against
keeping them on the device. `remove(restore=True)` makes them resident again (materialized once more).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from awpmi.materialization.weights import WeightStore


@dataclass
class StreamedParameters:
    model: nn.Module
    weights: WeightStore
    segments: Mapping[str, str]  # parameter name in the model → segment holding it
    calls: int = field(default=0, init=False)
    served_bytes: int = field(default=0, init=False)
    _by_module: dict = field(default_factory=dict, init=False)
    _handles: list = field(default_factory=list, init=False)

    def install(self) -> StreamedParameters:
        if self._handles:
            raise RuntimeError("already installed")
        by_module: dict[str, dict[str, str]] = defaultdict(dict)
        for name, segment in self.segments.items():
            owner, _, parameter = name.rpartition(".")
            module = self.model.get_submodule(owner)
            tensor = module._parameters.get(parameter)
            if tensor is None:
                raise KeyError(f"{name} is not a parameter of the model")
            info = self.weights.segment(segment)
            if (info.rows, *info.row_shape) != tuple(tensor.shape) or info.torch_dtype != tensor.dtype:
                raise ValueError(f"{name}: segment {segment} holds {info.dtype} {(info.rows, *info.row_shape)}, not {tensor.dtype} {tuple(tensor.shape)}")
            by_module[owner][parameter] = segment
        for owner, parameters in by_module.items():
            module = self.model.get_submodule(owner)
            for parameter in parameters:
                module._parameters[parameter] = None
            self._handles.append(module.register_forward_pre_hook(self._materialize(parameters)))
            self._handles.append(module.register_forward_hook(self._release(parameters), always_call=True))
        self._by_module = dict(by_module)
        return self

    def _materialize(self, parameters: dict[str, str]):
        def hook(module, args):
            self.calls += 1
            for parameter, segment in parameters.items():
                tensor = self.weights.rows(segment)
                self.served_bytes += tensor.numel() * tensor.element_size()
                module._parameters[parameter] = nn.Parameter(tensor, requires_grad=False)
            return None

        return hook

    def _release(self, parameters: dict[str, str]):
        def hook(module, args, output):
            for parameter in parameters:
                module._parameters[parameter] = None

        return hook

    def remove(self, restore: bool = True) -> None:
        """Remove the hooks; with `restore`, materialize every parameter once more and keep it resident."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if restore:
            with torch.inference_mode(False), torch.no_grad():
                for owner, parameters in self._by_module.items():
                    module = self.model.get_submodule(owner)
                    for parameter, segment in parameters.items():
                        module._parameters[parameter] = nn.Parameter(self.weights.rows(segment).clone(), requires_grad=False)
        self._by_module = {}
