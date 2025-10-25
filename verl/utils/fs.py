# -*- coding: utf-8 -*-
"""File-system agnostic IO APIs"""

import hashlib
import os
import shutil
import tempfile

try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from .hdfs_io import copy, exists, makedirs

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"


def is_non_local(path):
    return path.startswith(_HDFS_PREFIX)


def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def verify_copy(src: str, dest: str) -> bool:
    """
    verify the copy of src to dest by comparing their sizes and file structures.

    return:
        bool: True if the copy is verified, False otherwise.
    """
    if not os.path.exists(src):
        return False
    if not os.path.exists(dest):
        return False

    if os.path.isfile(src) != os.path.isfile(dest):
        return False

    if os.path.isfile(src):
        src_size = os.path.getsize(src)
        dest_size = os.path.getsize(dest)
        if src_size != dest_size:
            return False
        return True

    src_files = set()
    dest_files = set()

    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_root = os.path.join(dest, rel_path) if rel_path != "." else dest

        if not os.path.exists(dest_root):
            return False

        for entry in os.listdir(root):
            src_entry = os.path.join(root, entry)
            src_files.add(os.path.relpath(src_entry, src))

        for entry in os.listdir(dest_root):
            dest_entry = os.path.join(dest_root, entry)
            dest_files.add(os.path.relpath(dest_entry, dest))

    if src_files != dest_files:
        return False

    for rel_path in src_files:
        src_entry = os.path.join(src, rel_path)
        dest_entry = os.path.join(dest, rel_path)

        if os.path.isdir(src_entry) != os.path.isdir(dest_entry):
            return False

        if os.path.isfile(src_entry):
            src_size = os.path.getsize(src_entry)
            dest_size = os.path.getsize(dest_entry)
            if src_size != dest_size:
                return False

    return True

def copy_to_local(
    src: str, use_shm: bool = False
) -> str:

    if use_shm and isinstance(src, str) and not os.path.exists(src):
        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(src)
            if isinstance(resolved, str) and os.path.exists(resolved):
                src = resolved
        except ImportError:
            pass
        except Exception as e:
            print(f"WARNING: Failed to download model from Hugging Face: {e}")

    if use_shm:
        return copy_to_shm(src)
    return src

def copy_to_shm(src: str) -> str:
    shm_model_root: str = "/dev/shm/verl-cache/"
    src_abs: str = os.path.abspath(os.path.normpath(src))
    dest: str = os.path.join(shm_model_root, hashlib.md5(src_abs.encode("utf-8")).hexdigest())
    os.makedirs(dest, exist_ok=True)
    dest: str = os.path.join(dest, os.path.basename(src_abs))
    if os.path.exists(dest) and verify_copy(src, dest):
        print(
            f"[WARNING]: The memory model path {dest} already exists. If it is not you want, please clear it and "
            f"restart the task."
        )
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
    return dest

def local_mkdir_safe(path):
    """_summary_
    Thread-safe directory creation function that ensures the directory is created
    even if multiple processes attempt to create it simultaneously.

    Args:
        path (str): The path to create a directory at.
    """

    from filelock import FileLock

    if not os.path.isabs(path):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, path)

    # Using hash value of path as lock file name to avoid long file name
    lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
    lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

    try:
        with FileLock(lock_path, timeout=60):  # Add timeout
            # make a new dir
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to acquire lock for {path}: {e}")
        # Even if the lock is not acquired, try to create the directory
        os.makedirs(path, exist_ok=True)

    return path
