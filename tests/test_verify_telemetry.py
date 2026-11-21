from database import verify_telemetry
from database.telemetry import verify_jsonl_store
from benchmarks.freeze_evidence_bundle import build_evidence_bundle


def test_verify_telemetry_reports_unconfigured_mysql_without_connecting():
    result = verify_telemetry.verify("sqlite:///not-mysql.db")
    assert result == {"configured": False, "connected": False, "reason": "DATABASE_URL is not configured for local MySQL."}


def test_verify_jsonl_store_reports_operational_status():
    status = verify_jsonl_store()
    assert status["primary_store"] == "jsonl"
    assert status["status"] == "operational"
    assert status["total_records"] > 0
    assert "categories" in status


def test_freeze_evidence_bundle_contains_required_sections():
    bundle = build_evidence_bundle()
    assert "summary" in bundle
    assert "hardware_environment" in bundle
    assert "energy_capability_status" in bundle
    assert "dataset_readiness" in bundle
    assert "calibrated_profiles" in bundle
    assert "limitations_and_integrity_disclosures" in bundle
    assert bundle["summary"]["total_raw_telemetry_records"] > 0
