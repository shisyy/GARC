# Node 9.1: training-free relative evidence search

This is an **ideal full-mesh source diagnostic**, not reconstructed SplArt scene
performance. Renders are stylized, with two-color part shading. Geometry consists
of existing nearest-point/contact energy surrogates, not a collision-certified
SDF. The method provides no physical certificate or calibrated confidence.

## Frozen contract

The predictor sees no endpoints, source supervision, endpoint-trained checkpoint,
URDF limits, physical fractions or raw source dictionaries. A separate exporter
loads target-bearing geometry files but selects only fields 0:8 (posterior omitted)
and never indexes the target. Prediction runs exclusively on the strict whitelist.
The source split remains Articraft 13/6, even though no model is trained.

Coordinates use the **original observed states q=0 and q=1**. Outward distances d
query geometry at q=-d and q=1+d, matching the render plan. This intentionally
does not call the earlier virtual_profile (which uses observed q=0.1/0.9).
Consequently old endpoint-supervised baseline numbers are not comparable.
The fixed d grid is 129 positions from 0 through 2. Raw geometry covers [-2,3].
Out-of-support queries fail instead of clamping or choosing bounds from labels.

Frozen CLIP per-view embeddings and both observed anchors are retained. The
closed-minus-open text direction is generic and fixed. For each view, the signed
observed anchor difference determines outward direction. Candidate progress is
the image-feature difference from the corresponding observed anchor projected
onto this signed direction. Semantic cost is negative robust-scaled progress
plus robust-scaled absolute local slope, favoring advancement with slowing change.
This is a testable heuristic, not evidence that a plateau equals a hard limit.
10th/90th percentile scaling is within each trajectory and uses no labels.

Geometry cost is the unchanged unsupervised D2 total_energy channel, robust-scaled
per geometry/radius profile. Joint cost adds mean geometry and mean per-view
semantic cost with weight 1. The five preregistered controls use identical grids:
coordinate midpoint, geometry-only, semantic-only, joint, and joint with semantic
trajectories/anchors deterministically replaced by the next sorted source object
in the same split. All controls always produce a fallback point for both sides.

Abstention flags flat evidence, a search-boundary optimum, weak observed semantic
direction, opening (rather than closure) semantic evidence, broad/disconnected
near-optimum sets, or disagreement across views/geometry samples. Intervals are
the near-optimum hull when accepted and the full search grid when abstaining.
They are explicitly uncalibrated and may fail to contain a true endpoint outside
the finite grid. Full-mesh geometry roll-derived terms may have edge artifacts;
boundary optima therefore always abstain. Config values are frozen in
`configs/relative_search_v1.json` and the dataclass; no CLI tuning is exposed.

## Commands (parent selects authorized paths)

1. Build relative CLIP cache with `build_relative_clip_cache.py --help`.
2. Export: `python prepare_relative_search_inputs.py --source-index SOURCE_INDEX --semantic-index CLIP_INDEX --output-dir NEW_INPUT_DIR`
3. Freeze predictions: `python run_relative_search.py --inputs NEW_INPUT_DIR/index.json --output NEW_RESULTS/predictions.json`
4. Independent scoring: `python evaluate_relative_search.py --predictions NEW_RESULTS/predictions.json --source-index SOURCE_INDEX --output NEW_RESULTS/report.json`

Run steps 3 and 4 as separate processes. Step 3 writes all five controls before
evaluation, plus a SHA256 seal binding predictions and code hashes. Step 4 verifies
the seal/source binding before opening endpoint labels. Re-evaluation cannot
rewrite existing reports. The seal detects accidental changes; it is not an
external trusted timestamp or cryptographic proof against an authorized writer.

Primary score is unconditional validation joint mean-side NMAE; existing
side-NMAE and max-side target-relative p99 implementations are reused unchanged.
All objects/sides, including abstentions, enter the primary errors. Acceptance,
conditional errors (secondary only), empirical interval containment/width,
per-object errors and exact swap invariance are also reported. Partial pilot
exports require evaluator flag `--allow-train-smoke2`, are restricted to the first
two original IDs in sorted source_train order, and are explicitly marked incomplete.
Default evaluation requires every one of the 13/6 identities and every control;
dropping an object or method is an error. Predictor IDs are opaque SHA256 digests.

This first baseline exhaustively queries cached candidates. No trained LOOP module
has been added in this node. A useful semantic increment must first outperform
geometry-only and object-shuffled controls before a learned query policy is justified.
