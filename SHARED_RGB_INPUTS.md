# Fresh baseline and deletions from the same saved RGB

`rebase_shared_rgb_inputs.py` reads only the old strict target-free input cache
and the new shared-RGB acquisition. It never opens a raw source geometry file,
source label index, endpoint-trained checkpoint or evaluation report.

The common acquisition sidecar binds the new relative and part cache indices,
their identical renderer/encoder/builder provenance, the historic semantic
artifact hashes, and every one of the 258 saved RGB shards per object. The
exporter verifies all cache/shard file digests and exact roster before rebasing.
It preserves geometry, gauge coordinates, text direction and source-index hash;
only fresh joint/observed embeddings and their semantic artifact digest change.
The new strict input schema is unchanged. A separate `rebase_audit.json` binds
old/new input indices, both new cache indices and the acquisition sidecar.

    python rebase_shared_rgb_inputs.py --old-inputs OLD_STRICT_INDEX --acquisition-root SHARED_RGB_ROOT --output-dir NEW_STRICT_INPUTS

For the cache builder's two-object training smoke only, add
`--allow-train-smoke2`. Normal export requires the complete original13/6 roster.
The RGB verifier checks byte integrity and coverage; builder tests establish the
RGB tensor shape and that original/deletion features consume the same tensor.
The sidecar is a provenance receipt, not an independent physical certification.

After export, use the unchanged predictor runner with these new inputs, the new
`SHARED_RGB_ROOT/relative/index.json` baseline and
`SHARED_RGB_ROOT/part/index.json` interaction cache. Produce and verify all four
mode seals before independent endpoint evaluation. Check raw-mode equality
against `run_relative_search.py` **on the new inputs**. Historical semantic
predictions need not match a fresh acquisition. Geometry-only/coordinate-only
predictions should remain exact. The preregistered primary and all null controls
remain unchanged.
