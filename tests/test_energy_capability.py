from hardware.energy_capability import EnergyReading, joule_delta, probe, read_meter_export


def test_energy_probe_does_not_turn_a_proxy_into_measured_energy():
    report = probe()
    assert report["actual_energy_available"] is False
    assert report["energy_joules"] is None
    assert report["measurement_kind"] == "unavailable"


def test_energy_delta_requires_same_monotonic_actual_meter():
    assert joule_delta(EnergyReading("meter", 10), EnergyReading("meter", 13))["energy_joules"] == 3
    assert joule_delta(EnergyReading("meter-a", 10), EnergyReading("meter-b", 13))["energy_joules"] is None


def test_meter_export_requires_source_and_nonnegative_cumulative_joules(tmp_path):
    export = tmp_path / "meter.json"
    export.write_text('{"source":"wall meter","cumulative_joules":123.5}', encoding="utf-8")
    assert read_meter_export(export) == EnergyReading("wall meter", 123.5)
    export.write_text('{"source":"wall meter","cumulative_joules":-1}', encoding="utf-8")
    assert read_meter_export(export) is None


def test_energy_probe_reports_a_valid_configured_physical_meter(tmp_path):
    export = tmp_path / "meter.json"
    export.write_text('{"source":"wall meter","cumulative_joules":123.5}', encoding="utf-8")
    report = probe(export)
    assert report["actual_energy_available"] is True
    assert report["source"] == "wall meter"


def test_simulated_or_proxy_exports_are_never_accepted_as_physical_energy(tmp_path):
    export = tmp_path / "simulated.json"
    export.write_text('{"source":"cpu_time_proxy_simulated","measurement_kind":"estimated_proxy","cumulative_joules":123.5}', encoding="utf-8")
    assert read_meter_export(export) is None
    assert probe(export)["actual_energy_available"] is False
