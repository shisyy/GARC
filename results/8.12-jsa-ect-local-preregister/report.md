# Node 8.12 JSA-ECT result-free hardening

Status: implementation frozen locally; no remote execution, feature data, labels, source scores, Box data, or training were accessed.

The candidate uses one pooled domain-fixed-effect mean, a shared rank-one covariance axis, analytic energy-conserving scales, and a one-shot post-scale FWL refit. The transported residual is per-domain orthogonal to `[1, raw-z]`; donor reconstruction uses recipient-specific scales. Parallel and perpendicular energy blocks must each exceed the frozen relative identifiability floor, so affine, rank-one, zero-perpendicular, and width-one inputs fail closed.

Frozen invocation: `python -m scripts.run_jsa_ect_label_free`.

No feasibility outcome is claimed in this local preregistration report.
