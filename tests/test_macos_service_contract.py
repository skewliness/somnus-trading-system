from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v6_service_enables_layered_gate_but_keeps_unvalidated_gann_sizing_off() -> None:
    script = (ROOT / "scripts" / "macos_run_somnus_service.sh").read_text(encoding="utf-8")
    policy_section = script.split("configure_v6_policy()", 1)[1].split("configure_v6_instance()", 1)[0]
    demo_section = script.split("v6-execution)", 1)[1].split("v6-execution-live)", 1)[0]
    live_section = script.split("v6-execution-live)", 1)[1].split("v6-entry-gate-labeler)", 1)[0]

    assert "TS_V6_ENTRY_GATE_POLICY_V2_ENABLED=1" in policy_section
    assert "TS_V6_ENTRY_GATE_EVENT_RECORDING_ENABLED=1" in policy_section
    assert "TS_V6_GANN_POSITION_SIZING_ENABLED=0" in policy_section
    assert "TS_V6_GANN_POSITION_SIZING_ENABLED=1" not in policy_section
    assert "configure_v6_policy" in demo_section
    assert "configure_v6_policy" in live_section


def test_v6_live_loads_dedicated_credentials_and_uses_separate_state() -> None:
    script = (ROOT / "scripts" / "macos_run_somnus_service.sh").read_text(encoding="utf-8")
    instance_section = script.split("configure_v6_instance()", 1)[1].split('case "${1:-}"', 1)[0]

    assert ".env.v6_live" in instance_section
    assert 'state_dir="data/trading_system/v6_execution_live"' in instance_section
    assert 'export TS_V6_ORDER_MODE=live_order' in instance_section
    assert 'export TS_V6_EXECUTION_LOCK_TARGET="$state_dir/v6_execution"' in instance_section


def test_v6_live_installer_requires_readiness_before_launchd_install() -> None:
    installer = (ROOT / "scripts" / "macos_install_v6_live.sh").read_text(encoding="utf-8")

    readiness_index = installer.index("v6-live-readiness")
    install_index = installer.index("sudo install")
    assert readiness_index < install_index
    assert "com.somnus.v6-ledger-reconcile-live" in installer
    assert "com.somnus.v6-execution-live" in installer
    assert "com.somnus.telegram-v6-live" in installer


def test_v5_and_v6_telegram_services_are_push_only() -> None:
    script = (ROOT / "scripts" / "macos_run_somnus_service.sh").read_text(encoding="utf-8")
    for service, next_service in (
        ("telegram-v5)", "telegram-v6)"),
        ("telegram-v6)", "telegram-v6-live)"),
        ("telegram-v6-live)", "v5-trading)"),
    ):
        section = script.split(service, 1)[1].split(next_service, 1)[0]
        assert "TS_TELEGRAM_MONITOR_DISABLE_COMMANDS=1" in section
        assert "TS_TELEGRAM_MONITOR_DISABLE_INTEL_COMMANDS=1" in section
        assert "TS_TELEGRAM_MONITOR_DISABLE_SUMMARY=1" in section
        assert "TS_TELEGRAM_MONITOR_DISABLE_MACRO_REPORT=1" in section


def test_entry_gate_labeler_is_launchd_managed_and_cache_only() -> None:
    launch = (ROOT / "scripts" / "macos_launch_somnus_all.sh").read_text(encoding="utf-8")
    stop = (ROOT / "scripts" / "macos_stop_somnus_all.sh").read_text(encoding="utf-8")
    status = (ROOT / "scripts" / "macos_status_somnus_all.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "macos_run_somnus_service.sh").read_text(encoding="utf-8")
    labeler = (ROOT / "scripts" / "label_v6_entry_gate_events.py").read_text(encoding="utf-8")

    for script in (launch, stop, status):
        assert "com.somnus.v6-entry-gate-labeler" in script
    assert 'v6-entry-gate-labeler)' in runner
    assert "requests" not in labeler
    assert "urllib" not in labeler


def test_all_services_started_by_bulk_launcher_are_also_stopped_and_reported() -> None:
    launch = (ROOT / "scripts" / "macos_launch_somnus_all.sh").read_text(encoding="utf-8")
    stop = (ROOT / "scripts" / "macos_stop_somnus_all.sh").read_text(encoding="utf-8")
    status = (ROOT / "scripts" / "macos_status_somnus_all.sh").read_text(encoding="utf-8")
    labels = {
        line.strip()
        for line in launch.splitlines()
        if line.strip().startswith("com.somnus.")
    }

    assert labels
    for label in labels:
        assert label in stop
        assert label in status
