## Codebase

Working directory: D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node87_smarc

## Git Isolation

Work in the assigned experiment branch/worktree. Do not switch back to the main repository for implementation or evaluation.

## Research Idea

**ID**: 8.7
**Hypothesis**:
Mechanism: Semantic-Gated Mechanical Authored-Range Completion (SMARC) uses a frozen part-aware visual encoder only to gate bias-free mechanical residual experts that predict object-level authored joint range, then exactly projects that swap-invariant range around a frozen closure-selected analytic endpoint.
Hypothesis: Invisible hard stops encode functional design intent absent from collision fields, while restricting semantics to gate mechanical residuals recovers that intent without permitting a semantic category-to-range lookup.
Observable: On object- and family-disjoint Articraft/NJC validation SMARC beats displacement-only, category-frequency, global-range, mechanical-only, zero-image, and semantic/mechanical-shuffle controls by at least 15%; after freezing, it improves public Box a-d macro and worst-side NMAE by at least 10% versus both D2 and PILC, wins at least 3/4 episodes, and preserves state-swap error below 1e-6 before untouched e/f confirmation.
Conflicts: Nodes 3.1/6/8.1/8.2 show geometry-only terminal transitions are absent, nodes 4/4.1 show generic descriptor transfer is weak, and 8.5/8.6 show source geometry and uncertainty calibration collapse to priors; SMARC counters via authored-range supervision, token-level mechanical grounding, structural removal of semantic-only prediction, and mandatory shortcut nulls.

## Evaluation Info

- **Evaluation command (B_dev)**: `cd D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node87_smarc && python evaluate_public_endpoint_profiles.py --split public_box_af --run-name 8.7`
- **Evaluation command (B_test, do not use for routine experiments)**: `UNAVAILABLE_PROTECTED_B_TEST`
- **Dataset info**: Public Box a-f development episodes plus target-free first36 profiles; old sealed 18/9/9, B_test, and Full22 prohibited
- **Baseline score**: 0.154877
- **Current trunk score**: 0.154877

Use B_dev for final experiment scoring. Do NOT use B_test.

## Insights From Prior Experiments

- ROOT: Children findings: [1, done, score=0.3109] Original scratch SplArt accurately recovers axis and pivot but treats interior observations as endpoints: endpoint NMAE 0.310881, closed coverage 0, terminal validity 0. | [2, done, score=0.08573] D2-CEA reduced endpoint NMAE 72.42%, raised PSNR 3.72 dB and mobile IoU 46.74% with the base model hash unchanged. However all RMS/PCFG/OEC posthoc contact certificates abstained: closed coverage and terminal validity remain zero, showing the bottleneck is contact-surface representation rather than scalar search. | [3, done, score=0.08573] Children findings: [3.1, done, score=0.08573] View-balanced voxel fusion removed order/view-density bias, but observable surface contact and closure topology vote opposite sides; posthoc frozen-geometry terminal certification remains unidentifiable on Box. | [4, done, score=0.09515] Children findings: [4.1, done, score=0.1027] Deterministic visibility-censored Gaussianization slightly improved endpoint transfer but did not beat the NJC statistical prior; mesh-to-Gaussian appearance shift is not the sole bottleneck, and semantic/domain range priors remain dominant. | [5, pruned, score=0.1576] FECF only establ...
- 8: Children findings: [8.1, pruned, score=0.6882] Cross-state multi-radius consensus fails on representative Box-a: all 12 curves lack signed-gap zero crossings, consensus loses to D2 and a shuffled-geometry null, and uncertainty gating can only fall back to D2. [Pruned: Representative Box-a disproved cross-state/radius consensus on frozen penetrated profiles; the geometry shuffle null was better and gating yielded no gain.] | [8.2, pruned, score=0.343] The monotone hazard is ordered and semantics-sensitive on 36 target-free profiles, but Box-a has 12/12 signed-gap curves entirely nonpositive; hazard loses to frozen D2 and a channel-shuffle null, falsifying the free-to-terminal premise. [Pruned: Representative public Box-a falsified the mechanism: all 12 curves lack a free/contact crossing and the channel-shuffle null beats the proposed hazard.] | [8.3, pruned, score=0.4118] Projective consistency reduces exact36 target-domain cross-gauge variance by 37.34%, but fails domain transfer: on Box-a its worst-side NMAE and variance are worse than no-consistency, frozen D2, and coordinate controls. [Pruned: Target-free gauge stability improved in-domain but reversed under Box transfer and...

## Additional Context

Implement node 8.7 only in isolated worktree D:\workspace\projects\2026\09\05_splart_endpoint_extrapolation\worktrees\node87_smarc. This is real full execution. Phase A is a source-only feasibility gate; do not read public Box labels or any Box e/f/B_test/Full22 until source method and config are frozen. Server CASIA_98 conda Python is /data/yptang/workspace/20260720_SplArt/.conda/splart/bin/python with torch2.6+cu124, torchvision0.21, timm0.6.7 and PyTorch3D0.7.8. Source Articraft has no RGB, so deterministically render neutral-material alpha/silhouette canonical multiviews from meshes; reconstruct joins by replaying node4 parser and use only opaque SHA256 object_group_id, never category/archive strings as model input. First implement a ten-object AlexNet-ImageNet frozen-feature feasibility baseline because /home/yptang/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth exists. In parallel download official DINO ViT-B/16 checkpoint from https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth to /data1/public/yptang/splart-node87-smarc/models with SHA256 and load strictly through timm after stripping only expected classifier keys; do not call it DINOv2 or CLIP. Formal encoder is frozen DINO only if strict load and deterministic feature audit pass. SMARC semantics may only gate K=4 bias-free mechanical residual experts; no semantic-only additive path, category IDs, filenames, object IDs, or endpoint labels in inputs. Use exact swap-invariant range and exact anchored projection, continuous fixed gauge list, object/joint macro and family-disjoint audits. Pre-register one fixed config before scoring. Required source controls: global range, displacement-only, category-frequency audit-only, mechanical-only, zero-image (must equal mechanical-only structurally), semantic-shuffle within mechanical bins, mechanical-shuffle within semantic bins. Full must beat displacement/category/global/mechanical by >=15% on both Articraft and NJC and each shuffle must worsen >=20%; otherwise prune before Box. Start with tests plus the ten-object render/feature/join feasibility run, then shard full deterministic render/cache on verified-free GPUs 5/6/7 in durable tmux with CUBLAS env set before Python. Preserve exact PIDs/logs/hashes and protected_splits_read=[]. No seed search or hyperparameter sweep.

## Instructions

1. Understand the code before editing.
2. Implement the idea faithfully.
3. Run quick checks to ensure the new logic is active.
4. Iterate on implementation bugs.
5. Run the B_dev evaluation when credible.
6. Report Changes, Baseline vs Result, Score, and Insight. The score must be the absolute primary metric, not a delta.

Save results to `results/8.7-<brief-description>/`.
