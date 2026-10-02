"""Physical storage of weight pages (Phase 3), independent of certification and of any model.

The layers of a selective-materialization system, from the bottom (decision 0006):

  storage          `awpmi.storage`: where page bytes live and how they are read. Segments of
                   fixed-size rows in files or memory (`layout`), positioned reads with or
                   without the OS page cache (`fileio`), page stores that read only the
                   requested rows and count every byte (`store`), a page cache with a
                   replacement policy (`cache`), packs with a verifiable manifest (`pack`).
  transfer         `awpmi.streaming`: host → device movement (pinned staging, copy stream,
                   events, prefetch).
  materialization  `awpmi.materialization`: one API that turns "these rows of this segment" into
                   device tensors, through the cache, the store and the streamer.
  model adapters   `awpmi.models.*`: which segments a model's weights are, and when they are needed.
  policy           what to materialize: the Phase 1C certificate's contenders, the experts a
                   mixture-of-experts layer selects.
  certification    `awpmi.bounds`, `awpmi.certificate`: unchanged by where bytes come from.

Nothing in this package knows a model, a tensor name or an expert count.
"""
