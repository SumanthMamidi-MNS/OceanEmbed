"""Live nowcast: the released model on near-real-time satellite inputs.

See docs/research/live_nowcast.md.

* :mod:`~oceanembed.live.nrt`        near-real-time products: what is published, per-day download
* :mod:`~oceanembed.live.harmonise`  raw files -> canonical grid, the small live store
* :mod:`~oceanembed.live.update`     ``oceanembed live update`` / ``status`` (rolling window)
* :mod:`~oceanembed.live.shift`      input-shift check, near-real-time vs reprocessed inputs
* :mod:`~oceanembed.live.verify`     running verification against Argo and the operational analysis
"""
