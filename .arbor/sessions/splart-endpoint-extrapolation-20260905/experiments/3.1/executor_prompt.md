## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\jtcsr

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 3.1
**Hypothesis**:
Mechanism: Joint Topology-Contact Surface Refinement learns an explicit static-mobile interface SDF from public RGB-D and alternates endpoint-scalar projection with topology-contact consistency while distilling the frozen SplArt renders.
Hypothesis: The current failure is caused by optimizing endpoints against an inaccurate frozen Gaussian collision surface; a learned interface surface and alternating feasibility projection can create physically meaningful terminal support without sacrificing the successful D2 endpoint geometry.
Observable: On Box B_dev, endpoint NMAE stays at or below 0.085727 while closed coverage becomes nonzero with correct selective prediction, terminal validity becomes one, penetration does not increase, and endpoint rendering/articulation remain noninferior.
Conflicts: Nodes 2 and 3 showed posthoc RMS, PCFG, OEC, topology, and self-rendered certificates all abstain; this counters by changing the surface representation and training objective rather than relaxing any certificate gate.

## Evaluation Info

- **Evaluation command (B_dev)**: `python D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\jtcsr/endpoint_render_eval.py --prediction D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\jtcsr/prediction.json --sealed-evaluator-record <sealed-dev-record> --postbuild-seal <postbuild-seal> --output D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\.arbor\worktrees\jtcsr/score.json`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Box v4 proxy B_dev; public middle states only; sealed endpoints evaluator-only; B_test and Full22 protected/unavailable
- **Baseline score**: 0.31088139
- **Current trunk score**: 0.31088139

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- 3: Formal CCT produced informative topology (entropy 0.4489, confidence 0.7922, component agreement 0.75), but both endpoint supports failed physical recertification. A self-rendered 128px surface certificate likewise found high distributed contact but negative outside conflict gain, so posthoc reranking must abstain.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/3.1-<brief-description>/`.
