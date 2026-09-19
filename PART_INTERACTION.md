# Node 9.3: visible-part deletion interaction diagnostic

Four frozen modes: `raw`, `part_interaction`, `matched_partition`, and
`donor_interaction`. Every mode emits all five original node9.1 methods. Raw
delegates exactly to the unchanged predictor. Geometry-only and coordinate-only
also delegate unchanged under every mode. No old code/evaluator is modified.

Interaction features are **unnormalized** r=e_joint-e_static-e_mobile+e_blank
from unit CLIP components. The matched partition uses the same formula with
area-matched interleaved foreground masks. Only semantic image features/anchors
change; projected relative progress, slope cost, uniform view aggregation,
geometry cost and every old uncertainty rule remain the original formulas.
Zero projected residual change explicitly adds `zero_interaction_signal` and
abstains while keeping a finite unconditional fallback prediction.

`object_shuffle` first moves semantic ownership one cyclic object forward in the
same split, sorted by opaque ID. `donor_interaction` then moves ownership one
additional step. Cache artifact identity always matches the final owner. The
output records effective owners separately from the original semantic donor.

The new cache is bound to the exact base semantic index digest; each base row's
artifact hash must match the strict input's semantic provenance. Grids, roster,
encoder/renderer hashes and artifact digests are checked. No raw source index or
endpoint labels are runner arguments. First-two-source-train cache smoke can use
`load_interactions(cache_index, base_semantic_index, subset_values,
allow_train_smoke2=True)` without executing prediction or reading labels. Full
prediction requires the entire fixed13/6 roster and full cache.

Run after the cache builder finishes:

    python run_part_interaction.py --inputs STRICT_NODE91_INDEX --interaction-cache INTERACTION_INDEX --base-semantic-index BASE_CLIP_INDEX --output-dir NEW_RESULTS

All four prediction files and their seals are produced before a common
`all_modes.seal.json`; `verify_all_modes(Path(NEW_RESULTS))` checks all of them and
is called automatically on completion. Parent must verify this manifest before
launching the unchanged independent evaluator for each mode:

    python evaluate_relative_search.py --predictions NEW_RESULTS/MODE/predictions.json --source-index AUTHORIZED_SOURCE_INDEX --output NEW_RESULTS/MODE/report.json

Primary remains **part_interaction joint mean-side NMAE**. Semantic-only is a
secondary representation diagnostic. Report every mode/control, unconditional
mean/worst/p99 including abstentions, and coverage. This is a full-mesh stylized
source diagnostic, not reconstructed-scene accuracy or physical causality.
Visible pixel deletion does not reveal occluded surfaces or hidden stops. No
learned LOOP component or publishable novelty is established by implementation.
