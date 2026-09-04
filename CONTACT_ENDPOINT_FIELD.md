# Contact-Feasible Endpoint Field prototype

This node predicts motion limits in the relabelled coordinate system where the
two observed interior states are `0` and `1`. It receives reconstructed static
and mobile Gaussian geometry plus SplArt's predicted relative screw motion. It
never receives the original articulation fractions, URDF limits, endpoint
images, or endpoint labels.

For each extrapolation direction, the module scans a fixed scalar interval and
computes four geometry-only terms:

1. surface-contact energy at the candidate configuration;
2. interpenetration energy at the candidate;
3. terminal-support energy requiring penetration to rise just beyond the
   candidate in the outward direction;
4. an inside-feasibility energy requiring the neighboring inward
   configuration to remain collision-free.

The two endpoint estimates are soft argmins of these differentiable fields.
The physical certificate checks contact proximity, one-sided support rise,
inside feasibility, posterior concentration, and scan-boundary selection. If a
stop is not identified, its public value is `NaN`; the diagnostic soft estimate
is retained separately. This avoids hallucinating a joint limit in free space.

The closed-end label uses a broad-versus-localized contact-area signature. If
the two contacts are physically symmetric, the label is `unknown` even when
both limit locations are identifiable. This is intentional: geometry alone
does not always contain enough information to assign semantic closure.

The current implementation uses spherical Gaussian envelopes and either dense
small-problem pairs or a caller-supplied deterministic broad-phase pair list.
It is an analytic CPU prototype only. It does not read B_dev metrics and is not
yet wired into training; candidate-versus-control evaluation must wait until
the new from-scratch extrapolation baseline and sealed benchmark are complete.
