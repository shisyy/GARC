# Idea Tree

**Baseline**: N/A | **Trunk**: N/A

## ROOT: Given two interior articulation states relabelled 0 and 1, predict the fully closed endpoint and the opposite physical joint limit, and reconstruct held-out endpoint views without access to absolute interior fractions or true limits. [DONE]

### 1: Mechanism: Middle-State Episode Reparameterization builds two interior multi-view states, erases their absolute joint fractions from model-facing metadata, and evaluates predicted closed/limit endpoints on held-out endpoint cameras.
Hypothesis: The current task appears successful because training states equal the URDF limits; moving both observations strictly inside the range will expose true extrapolation error and establish an honest new baseline.
Observable: A sealed Dev3 proxy passes leakage and geometry audits, and unmodified scratch SplArt produces endpoint-limit, endpoint-render, depth, IoU, and articulation baseline metrics.
Conflicts: none - this changes the benchmark assumption rather than retreading a failed optimization mechanism. [PENDING]

### 2: Mechanism: Contact-Feasible Endpoint Field scans the learned screw trajectory with differentiable static/mobile contact, penetration, and terminal-support energies, predicting two latent endpoint scalars plus a closed-end label and uncertainty interval.
Hypothesis: Physical endpoints are extrema of the collision-free configuration interval; broad closure contact and localized stop contact provide different signatures that can extrapolate beyond two interior observations.
Observable: On sealed middle-state Dev3, normalized endpoint error and closed-end accuracy improve over scratch SplArt while endpoint PSNR, SSIM, LPIPS, depth, IoUs, and articulation remain noninferior.
Conflicts: prior IKC point-to-point experiments optimized consistency between observed endpoint states rather than inferring unseen limits; this node uses them only as implementation lessons and adds a latent feasible interval with no limit labels. [PENDING]
