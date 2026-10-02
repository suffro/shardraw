"""Materialization layer (Phase 3): "these rows of this segment" → device tensors, every byte counted.

`MaterializationBackend` composes a page store, a page cache and a streamer behind one
call. `WeightStore` gives typed views of named weights over it; `ExpertStore` groups the
expert-sliced weights of mixture-of-experts layers. None of them knows a model: segment
names, group keys and expert counts come from the model adapter or the pack.
"""
