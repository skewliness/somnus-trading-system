from __future__ import annotations

import os
import time

from scripts.maintain_kline_caches import clean_stale_execution_request_locks


def test_clean_stale_execution_request_locks_keeps_recent_locks(tmp_path) -> None:
    old_lock = tmp_path / ".execution_request_old.json.lock"
    recent_lock = tmp_path / ".execution_request_recent.json.lock"
    request_json = tmp_path / "execution_request_old.json"
    old_lock.write_text("")
    recent_lock.write_text("")
    request_json.write_text("{}")

    old_time = time.time() - 3 * 86400
    os.utime(old_lock, (old_time, old_time))

    summary = clean_stale_execution_request_locks(tmp_path, ttl_days=2, dry_run=False)

    assert summary["files_seen"] == 2
    assert summary["files_deleted"] == 1
    assert not old_lock.exists()
    assert recent_lock.exists()
    assert request_json.exists()
