# Experiment 8.14

**Hypothesis**: Mechanism: Gauge-Equivariant Looped Primal-Dual Transformer (GLPDT) tokenizes each side's ordered multi-radius D2 profile and recurrently applies one pre-norm input-recalled Transformer block for four shared-weight loops to refine a positive endpoint-distance state and a terminal-violation state, with one shared side function and deep supervision at every loop.
Hypothesis: One-pass heads and distribution transports fail because they compress or warp profiles that contain no clean boundary crossing; recurrent bounded residual refinement can accumulate weak geometric evidence without PKSRT-style tail inflation, while shared side weights and input recall prevent observation-order and fixed-point collapse.
Observable: On object-disjoint source B_dev before any Box or protected evaluation, GLPDT lowers absolute endpoint NMAE versus frozen D2 and a parameter-matched one-pass Transformer, zero-geometry, coordinate-only and profile-shuffle controls; exact swap error is <=1e-6, loop residual decreases, and prediction p99 inflation is <=1.25.
Conflicts: Pruned [8.1]/[8.2] found no usable profile crossing and [8.13] inflated three tails; this node assumes no explicit crossing and instead performs contractive residual inference around the frozen D2 anchor, with fail-closed null and tail gates.

**Score**: 0.327601134777069

**Insight**: Tied recurrent refinement cuts source-validation mean-side NMAE from 0.695073 to 0.327601 with exact swap error 0 and monotonically decreasing loop residuals, validating iterative evidence accumulation. However only 6/12 validation sides are reachable under the fixed additive cap and prediction/anchor p99 ratio is 32.8629, so the next bottleneck is the additive primal parameterization and tiny-anchor tail.

**Result**: PASS_SOURCE_MEAN: GLPDT improved mean-side source B_dev NMAE 52.87% versus its same-protocol D2 anchor. Artifact SHA256 report=8da4dcf585cb150fc789b7cbbee00b13713641aa89549b3c5fe25971b38a0cc4 checkpoint=f35a8394615c92aba33643f93a2d9e8e2f55173703bf9de191ea2d762cc8cf61; protected_splits_read=[]; fixed-cap tail gate failed.
