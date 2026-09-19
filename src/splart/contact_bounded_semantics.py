"""Frozen geometry-bounded, closure-only semantic localization; no labels."""
import torch

from splart.relative_search import METHODS, predict, validate_input
from splart.geometry_first import semantic_role, _candidate_verification, _validate_donor
from splart.contact_calibration import first_event

MODES = ("raw", "bounded", "donor_semantics", "shuffled_verification")


def swap_contact(contact):
    """Prediction-only swapped view; the observed contact atlas is invariant."""
    return dict(contact, sides=list(reversed(contact["sides"])), coordinates=contact["coordinates"].flip(0))


def bounded_predict(value, method, contact, calibration, mode="bounded", semantic_donor=None, route_donor=None):
    if mode not in MODES or method not in METHODS:
        raise ValueError("unknown frozen mode/method")
    if mode == "raw":
        return predict(value, method, semantic_donor)
    if method in ("coordinate_only", "semantic_only"):
        return predict(value, method)
    validate_input(value)
    identity = value["split"], value["object_id"]
    if any((item["split"], item["object_id"]) != identity for item in (contact, calibration)):
        raise ValueError("contact/calibration identity mismatch")
    if not torch.equal(contact["coordinates"], value["coordinates"]):
        raise ValueError("contact/input coordinates mismatch")
    if not isinstance(calibration["calibration_passed"], bool) or calibration["enumeration_complete"] is not True or calibration["exact_swap_passed"] is not True:
        raise ValueError("complete verified calibration required")
    owner = value
    if method == "object_shuffle" or (mode == "donor_semantics" and method == "joint"):
        owner = _validate_donor(value, semantic_donor, "semantic control")
    role_owner = owner
    if mode == "shuffled_verification" and method in ("joint", "object_shuffle"):
        role_owner = _validate_donor(value, route_donor, "routing control")
    fallback = predict(value, "geometry_only")
    sides = []
    for side in range(2):
        d = value["coordinates"][side]
        event = first_event(contact, side)
        passed = calibration["calibration_passed"]
        available = passed and event is not None and event > 1
        candidates = list(range(1, event)) if available else []
        diagnostic = {"geometric_confirmed": bool(available), "geometric_available": bool(available),
                      "calibration_passed": passed, "first_event_index": event, "candidate_indices": candidates,
                      "prefix": [1, event - 1] if available else None,
                      "role": "not_evaluated", "role_eligible": False, "semantic_activation": False,
                      "semantic_verified": False, "selection_changed": False, "fallback_reason": None,
                      "semantic_owner": owner["object_id"], "routing_owner": role_owner["object_id"],
                      "verification": [], "physical_certification": False}
        if not available:
            reason = "midpoint_calibration_failed" if not passed else ("no_new_surface_contact" if event is None else "no_outward_precontact_choice")
            result = dict(fallback["sides"][side])
            result["reasons"] = sorted(set(result["reasons"] + [reason]))
            result["abstain"] = True
            result["evidence_interval"] = [float(d[0]), float(d[-1])]
            diagnostic["fallback_reason"] = reason
            diagnostic["selected_index"] = int((d - result["distance"]).abs().argmin())
        else:
            geometric = candidates[-1]
            selected = geometric
            reasons = []
            cost = (d - d[geometric]).abs().double()
            cost[0] = float(cost.max()) + 1
            cost[event:] = float(cost.max()) + 1
            if method != "geometry_only":
                role = semantic_role(role_owner, side)
                diagnostic.update(role)
                if role["role"] != "closure_like":
                    reason = "unknown_semantic_role" if role["role"] == "unknown" else "opening_role_not_localized"
                    reasons.append(reason)
                    diagnostic["fallback_reason"] = reason
                else:
                    diagnostic["semantic_activation"] = True
                    verification = _candidate_verification(owner, side, candidates, 1)
                    diagnostic["verification"] = verification
                    verified = [r for r in verification if r["verified"]]
                    if not verified:
                        reasons.append("semantic_verification_failed")
                        diagnostic["fallback_reason"] = "semantic_verification_failed"
                    else:
                        selected = min(verified, key=lambda r: (-r["median_margin"], abs(float(d[r["index"]] - d[geometric])), r["index"]))["index"]
                        diagnostic["semantic_verified"] = True
                        diagnostic["selection_changed"] = selected != geometric
                        cost = torch.full_like(cost, 1 + max(abs(r["median_margin"]) for r in verification))
                        for row in verified:
                            cost[row["index"]] = -row["median_margin"]
            hull = [float(d[candidates[0]]), float(d[candidates[-1]])]
            result = {"distance": float(d[selected]), "abstain": bool(reasons), "reasons": sorted(set(reasons)),
                      "evidence_interval": [float(d[0]), float(d[-1])] if reasons else hull,
                      "near_optimum_hull": hull, "component_optima": [float(d[geometric])], "cost": cost.tolist()}
            diagnostic["selected_index"] = selected
        result["contact_bounded"] = diagnostic
        sides.append(result)
    return {"method": method, "object_id": value["object_id"], "split": value["split"],
            "distance": [s["distance"] for s in sides], "sides": sides,
            "uncertainty": "heuristic_not_calibrated_not_physical_certification"}
