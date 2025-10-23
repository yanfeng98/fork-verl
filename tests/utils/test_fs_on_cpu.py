import os
from pathlib import Path

import verl.utils.fs as fs


def test_copy_from_hdfs_with_mocks(tmp_path, monkeypatch):
    # Mock HDFS dependencies
    monkeypatch.setattr(fs, "is_non_local", lambda path: True)

    # side_effect will simulate the copy by creating parent dirs + empty file
    def fake_copy(src: str, dst: str, *args, **kwargs):
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(b"")  # touch an empty file

    monkeypatch.setattr(fs, "copy", fake_copy)  # Mock actual HDFS copy

    # Test parameters
    test_cache = tmp_path / "cache"
    hdfs_path = "hdfs://test/path/file.txt"

    # Test initial copy
    local_path = fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    expected_path = os.path.join(test_cache, fs.md5_encode(hdfs_path), os.path.basename(hdfs_path))
    assert local_path == expected_path
    assert os.path.exists(local_path)


def test_always_recopy_flag(tmp_path, monkeypatch):
    # Mock HDFS dependencies
    monkeypatch.setattr(fs, "is_non_local", lambda path: True)

    copy_call_count = 0

    def fake_copy(src: str, dst: str, *args, **kwargs):
        nonlocal copy_call_count
        copy_call_count += 1
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(b"")

    monkeypatch.setattr(fs, "copy", fake_copy)  # Mock actual HDFS copy

    test_cache = tmp_path / "cache"
    hdfs_path = "hdfs://test/path/file.txt"

    # Initial copy (always_recopy=False)
    fs.copy_to_local(hdfs_path)
    assert copy_call_count == 1

    # Force recopy (always_recopy=True)
    fs.copy_to_local(hdfs_path)
    assert copy_call_count == 2

    # Subsequent normal call (always_recopy=False)
    fs.copy_to_local(hdfs_path)
    assert copy_call_count == 2  # Should not increment
