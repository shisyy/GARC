# Node 9.2: observed-motion reliability aggregation

One frozen change to node9.1: replace the equal view average of semantic cost
with a label-free weighted average. Geometry, signed relative progress, slope
formula, numerical scale, near-optimum tolerance, thresholds, and all uncertainty
checks remain the original implementation. No learned LOOP policy is claimed.

For view v, the shared two-side weight is

    signal_v = ((anchor1_v - anchor0_v) dot closed_open_text)^2
    noise_v = mean_{both sides, interior positions}((second_difference(image_v dot text))^2)
    weight_v = normalize_views(signal_v / (noise_v + float64_epsilon))

Compute in float64, reducing each side first and then adding the two side noise
means. This preserves exact input-order reversal. If every observed projection
is exactly zero, use uniform weights and keep the original weak-direction
abstention. There is no learned/tuned smoothing, temperature, threshold or floor.

Preregistered modes, all emitted in one run:

- `uniform`: delegate to node9.1 exactly, including every output field.
- `reliability`: signal/noise weights above.
- `inverse_noise_only`: remove the signal numerator.
- `shuffled_weights`: take reliability weights from the next sorted opaque object
  ID within the same source split, without changing recipient semantic evidence.

Each mode emits all five original methods. For `object_shuffle`, semantic
trajectories and anchors belong to the usual semantic donor, so reliability
weights normally belong to that same donor. In `shuffled_weights`, the weights
come from the next object after the semantic donor. This preserves the definition
of the original semantic-shuffle control. All original unweighted component
optima and weak/opening checks remain in the abstention rules: downweighting a
view never removes its disagreement flag. Zero accepted coverage can therefore
remain even when point estimates improve.

Reuse the immutable node9.1 19-object input index; no labels, meshes, prompts,
rendering, downloads or training are needed:

    python run_view_reliability.py --inputs /data1/public/yptang/splart-node91-relative-search/inputs-articraft-v1/index.json --output-dir NEW_RESULT_DIR

This command requires the fixed 13/6 roster and seals all four mode outputs.
Only after it succeeds and `NEW_RESULT_DIR/all_modes.seal.json` exists, run the
unchanged independent evaluator for each mode:

    python evaluate_relative_search.py --predictions NEW_RESULT_DIR/MODE/predictions.json --source-index AUTHORIZED_SOURCE_INDEX --output NEW_RESULT_DIR/MODE/report.json

The parent coordinates independent scoring. Every output includes code hashes,
the mode roster, per-object weights, signal/noise diagnostics and effective view
count (1/sum(weights^2)). All four unconditional mean/worst/p99 scores must be
reported, including abstained sides, without choosing a favorable mode after
evaluation. Full-mesh stylized source diagnostic and uncertainty limitations
from `RELATIVE_SEARCH.md` still apply.
