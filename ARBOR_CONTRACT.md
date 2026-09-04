# Arbor contract: intermediate-state endpoint extrapolation

- Target: this isolated worktree on branch `codex/endpoint-extrapolation-baseline`.
- Baseline source: unmodified SplArt commit `2e5e286b3ffc027d37c9c9da1a5dc87d18301849` plus benchmark-only data plumbing.
- Task: reconstruct an articulated object from two interior articulation states and predict the two physically valid motion endpoints: the fully closed endpoint and the opposite joint-limit endpoint.
- Observation convention: the two observed interior states are relabelled `0` and `1`; their absolute positions within the full joint range and the URDF limits are not model inputs.
- Initial proxy B_dev: fixed `100247-Box`, `102016-USB`, and `103042-Window`, covering revolute and prismatic joints. Large assets live on CASIA server 98.
- B_test: a fixed disjoint scene set, hidden from routine iteration. Full22 remains final verification only.
- Baseline: train original SplArt from random initialization on the two middle-state observations, then measure endpoint prediction and endpoint novel-view reconstruction without loading any author or prior-task checkpoint.
- Primary objective: minimize normalized endpoint-limit error while correctly identifying the closed endpoint. A method is ineligible if endpoint rendering, depth, segmentation, or articulation validity regresses against the new-task baseline.
- Endpoint metrics: lower/upper normalized limit MAE, closed-end identification accuracy, terminal-contact/penetration validity, endpoint PSNR, SSIM, LPIPS, depth MAE, static/mobile/background IoU, mIoU, and articulation errors.
- Allowed edits: benchmark data generator, dataparser, model, physics endpoint module, training configuration, launchers, validators, tests, and reports.
- Protected: original datasets and transforms, URDF/ground-truth limits, held-out endpoint images, B_test/Full22 membership, official metric definitions, and existing SplArt-CVPR uncommitted work.
- Prohibited: providing the hidden interior fractions or true joint limits to the model; seed/checkpoint/view/result selection; evaluator changes; test-derived thresholds; author checkpoints; killing unrelated processes.
- Compute: real from-scratch A6000 experiments are authorized only on live-free GPUs after independent resource and provenance checks.
- Run mode: real, automatic, novelty/performance mixed. Initial cycle cap 12; first produce a sealed proxy dataset and baseline before optimizing physics.

The benchmark builder may use ground-truth kinematics only to generate and score data. Model-facing files must contain only relabelled observation states and must pass a leakage audit.
