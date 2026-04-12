"""
agent/tools/filesystem.py

Filesystem tools for AMADEUS.
All tools validate paths through security.validators before any I/O.
Destructive operations (delete, move, overwrite) use the HITL confirmation
registry and return CONFIRMATION_SENTINEL until the user approves.
"""
import shutil
from pathlib import Path

from langchain_core.tools import tool

from agent.hitl import (
    CONFIRMATION_SENTINEL,
    is_confirmed,
    register_pending,
)
from security.validators import validate_path, PathSecurityError


# ─── Read-Only Tools ──────────────────────────────────────────────────────────


@tool
def list_directory(path: str) -> str:
    """
    List the contents of a directory, showing file names, types, and sizes.
    Use this to explore what files and folders exist at a given path.
    Example: list_directory("C:/Users/me/Documents")
    """
    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not validated.exists():
        return f"Error: Path does not exist: {validated}"
    if not validated.is_dir():
        return f"Error: '{validated}' is a file, not a directory. Use read_file to read it."

    try:
        entries = sorted(validated.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return f"Error: Permission denied reading directory: {validated}"

    if not entries:
        return f"Directory is empty: {validated}"

    lines = [f"Contents of '{validated}' ({len(entries)} items):"]
    for item in entries:
        kind = "FILE" if item.is_file() else "DIR "
        size_str = ""
        if item.is_file():
            try:
                size_str = f"  ({item.stat().st_size:,} bytes)"
            except OSError:
                size_str = "  (size unknown)"
        lines.append(f"  [{kind}] {item.name}{size_str}")

    return "\n".join(lines)


@tool
def read_file(path: str) -> str:
    """
    Read and return the text content of a file.
    Only works on plain text files (e.g., .txt, .py, .json, .md, .csv, .log).
    For Excel files, use read_excel instead.
    Respects a maximum file size limit configured in the environment.
    Example: read_file("C:/Users/me/notes.txt")
    """
    from security.validators import get_max_file_size_bytes

    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not validated.exists():
        return f"Error: File does not exist: {validated}"
    if not validated.is_file():
        return f"Error: '{validated}' is a directory. Use list_directory to explore it."

    max_bytes = get_max_file_size_bytes()
    try:
        size = validated.stat().st_size
    except OSError as exc:
        return f"Error accessing file: {exc}"

    if size > max_bytes:
        return (
            f"Error: File is too large to read directly ({size:,} bytes). "
            f"Maximum allowed: {max_bytes:,} bytes ({max_bytes // (1024*1024)} MB). "
            f"Consider using a data analysis tool if this is a CSV/Excel file."
        )

    try:
        content = validated.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        line_count = len(lines)
        truncated = ""
        # Truncate very long outputs to avoid overwhelming the LLM context
        if line_count > 500:
            content = "\n".join(lines[:500])
            truncated = f"\n\n[Output truncated: showing first 500 of {line_count} lines]"
        return f"=== {validated.name} ({size:,} bytes, {line_count} lines) ===\n{content}{truncated}"
    except Exception as exc:
        return f"Error reading file: {exc}"


# ─── Mutating Tools (no HITL needed) ─────────────────────────────────────────


@tool
def create_directory(path: str) -> str:
    """
    Create a new directory (and any missing parent directories) at the given path.
    Safe to call if the directory already exists.
    Example: create_directory("C:/Users/me/Documents/new_project")
    """
    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    try:
        validated.mkdir(parents=True, exist_ok=True)
        return f"Directory created (or already exists): {validated}"
    except PermissionError:
        return f"Error: Permission denied creating directory: {validated}"
    except Exception as exc:
        return f"Error creating directory: {exc}"


# ─── Destructive Tools (HITL required) ───────────────────────────────────────


@tool
def delete_file(path: str) -> str:
    """
    Delete a file or directory (including all its contents) at the given path.
    WARNING: This is a DESTRUCTIVE, IRREVERSIBLE operation.
    User confirmation is required before the deletion is executed.
    Example: delete_file("C:/Users/me/Desktop/old_report.txt")
    """
    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not validated.exists():
        return f"Error: Nothing to delete - path does not exist: {validated}"

    action_key = f"delete:{validated}"

    if not is_confirmed(action_key):
        item_type = "directory and all its contents" if validated.is_dir() else "file"
        register_pending(
            key=action_key,
            description=f"Permanently delete {item_type} '{validated.name}'?",
            action_type="delete_file",
            params={
                "path": str(validated),
                "type": "directory" if validated.is_dir() else "file",
            },
        )
        return CONFIRMATION_SENTINEL

    # User confirmed - proceed with deletion
    try:
        if validated.is_dir():
            shutil.rmtree(validated)
        else:
            validated.unlink()
        return f"Successfully deleted: {validated}"
    except PermissionError:
        return f"Error: Permission denied deleting: {validated}"
    except Exception as exc:
        return f"Error deleting '{validated}': {exc}"


@tool
def move_file(source: str, destination: str) -> str:
    """
    Move a file or directory from source path to destination path.
    The destination can be a new filename or a target directory.
    WARNING: This is a DESTRUCTIVE operation requiring user confirmation.
    If destination exists, it will be overwritten.
    Example: move_file("C:/Users/me/Desktop/report.txt", "C:/Users/me/Documents/report.txt")
    """
    try:
        src = validate_path(source)
    except PathSecurityError as exc:
        return f"Security error (source): {exc}"

    # Validate the _parent_ of the destination so it doesn't need to exist yet
    dst_raw = Path(destination)
    try:
        dst_parent = validate_path(str(dst_raw.parent))
    except PathSecurityError as exc:
        return f"Security error (destination): {exc}"

    if not src.exists():
        return f"Error: Source does not exist: {src}"

    dst = dst_parent / dst_raw.name
    action_key = f"move:{src}:{dst}"

    if not is_confirmed(action_key):
        register_pending(
            key=action_key,
            description=f"Move '{src.name}' to '{dst}'?",
            action_type="move_file",
            params={"source": str(src), "destination": str(dst)},
        )
        return CONFIRMATION_SENTINEL

    try:
        shutil.move(str(src), str(dst))
        return f"Moved:\n  From: {src}\n  To:   {dst}"
    except PermissionError:
        return f"Error: Permission denied during move."
    except Exception as exc:
        return f"Error moving file: {exc}"


@tool
def copy_file(source: str, destination: str) -> str:
    """
    Copy a file or directory from source to destination.
    If the destination already exists, confirmation is required (overwrite).
    If the destination does not exist, the copy proceeds without confirmation.
    Example: copy_file("C:/Users/me/report.xlsx", "C:/Users/me/backup/report_backup.xlsx")
    """
    try:
        src = validate_path(source)
    except PathSecurityError as exc:
        return f"Security error (source): {exc}"

    dst_raw = Path(destination)
    try:
        dst_parent = validate_path(str(dst_raw.parent))
    except PathSecurityError as exc:
        return f"Security error (destination): {exc}"

    if not src.exists():
        return f"Error: Source does not exist: {src}"

    dst = dst_parent / dst_raw.name

    # Only require HITL confirmation if destination exists (overwrite scenario)
    if dst.exists():
        action_key = f"copy_overwrite:{src}:{dst}"
        if not is_confirmed(action_key):
            register_pending(
                key=action_key,
                description=f"Overwrite existing '{dst.name}' at '{dst.parent}' with a copy of '{src.name}'?",
                action_type="copy_file",
                params={"source": str(src), "destination": str(dst), "overwrite": "yes"},
            )
            return CONFIRMATION_SENTINEL

    try:
        if src.is_dir():
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src), str(dst))
        return f"Copied:\n  From: {src}\n  To:   {dst}"
    except PermissionError:
        return f"Error: Permission denied during copy."
    except Exception as exc:
        return f"Error copying: {exc}"
