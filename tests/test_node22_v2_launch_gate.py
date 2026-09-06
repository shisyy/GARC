from pathlib import Path

from audit_node22_v2_launch_gate import audit


ROOT = Path(__file__).resolve().parents[1]


def test_current_repository_fails_closed_without_forensic_receipts():
    result = audit(ROOT)
    assert result["status"] == "FAIL"
    assert result["launch_authorized"] is False
    assert result["private_inputs_opened"] is False
    assert result["protected_splits_read"] == []
    assert result["targets_read"] == []
    assert "V1-TERMINAL-EVIDENCE-MISSING" in result["failures"]
    assert "V2-INDEPENDENT-P0-NOT-AUTHORIZED" in result["failures"]


def test_current_repository_catches_metric_semantics_before_launch():
    result = audit(ROOT)
    assert "NMAE-MULTIPLIER-MISSING" not in result["failures"]
    assert "PRIMARY-OBJECT-SCORE-NOT-DUAL-SIDE-MAX" not in result["failures"]
    assert "INTERVAL-WIDTH-NOT-NMAE-NORMALIZED" not in result["failures"]
    assert "NMAE-SEMANTIC-SYNTHETIC-TEST-MISSING" not in result["failures"]
    assert "NORMALIZATION-FORMULA-VALUE-NOT-ENFORCED" in result["failures"]
    assert "CONSTANT-WIDTH-NOT-NMAE-NORMALIZED" in result["failures"]


def test_current_repository_catches_incomplete_recovery_and_cvpr_plan():
    failures = set(audit(ROOT)["failures"])
    assert {
        "V3-NOT-FORBIDDEN",
        "NEW-OUTPUT-ROOT-NOT-REQUIRED",
        "V2-RUNNER-HASH-NOT-FROZEN",
        "LEARNED-SCALE-ABLATION-MISSING",
        "BASELINE-EXPORTS-NOT-FROZEN-READY",
        "D2-EXPORTER-EXACT36-NOT-ENFORCED",
        "D2-EXPORTER-PRIVATE-FIELDS-NOT-REJECTED",
        "D2-EXPORTER-OBJECT-ID-PATH-UNSAFE",
        "V2-AUTHORIZATION-STATUS-NOT-ENFORCED",
        "TRUTH-OBJECT-DISJOINTNESS-NOT-ENFORCED",
    }.issubset(failures)
