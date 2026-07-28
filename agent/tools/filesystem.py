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


@tool
def write_file(path: str, content: str) -> str:
    """
    Create a new file with the given content, or overwrite an existing one.
    Use this for full rewrites (e.g., generating a new script) or when the
    user explicitly wants to replace an entire file's contents.
    For small, targeted changes to an existing file, prefer `edit_file`
    instead — it's safer because it only touches the specific part that changed.
    WARNING: overwriting an EXISTING file is DESTRUCTIVE and requires user
    confirmation. Creating a brand-new file does not.
    Example: write_file("C:/Proyectos/mi-app/utils.py", "def suma(a, b):\\n    return a + b\\n")
    """
    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if validated.is_dir():
        return f"Error: '{validated}' is a directory, not a file."

    file_exists = validated.exists()

    if file_exists:
        action_key = f"write:{validated}:{hash(content)}"
        if not is_confirmed(action_key):
            try:
                preview = validated.read_text(encoding="utf-8", errors="replace")
            except Exception:
                preview = "(no se pudo leer el contenido actual)"
            register_pending(
                key=action_key,
                description=(
                    f"Sobrescribir el archivo '{validated.name}' "
                    f"({len(preview.splitlines())} líneas actuales → "
                    f"{len(content.splitlines())} líneas nuevas)?"
                ),
                action_type="write_file",
                params={"path": str(validated), "content_preview": content[:500]},
            )
            return CONFIRMATION_SENTINEL

    try:
        validated.parent.mkdir(parents=True, exist_ok=True)
        validated.write_text(content, encoding="utf-8")
        verb = "sobrescrito" if file_exists else "creado"
        return f"Archivo {verb} correctamente: {validated} ({len(content.splitlines())} líneas)"
    except PermissionError:
        return f"Error: Permission denied writing to: {validated}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool
def edit_file(path: str, find: str, replace: str, count: int = 1) -> str:
    """
    Make a targeted edit to an existing file: replaces an exact snippet of
    text ('find') with new text ('replace'), leaving the rest of the file
    untouched. Prefer this over `write_file` for small changes (fixing a
    bug, renaming a variable, adding a parameter) since it's safer and lets
    the user review exactly what changes.
    'find' must match the file's content EXACTLY (including whitespace/indentation) —
    read the file first with `read_file` to copy the exact snippet.
    'count' limits how many occurrences to replace (default 1, use -1 for all).
    WARNING: this is a DESTRUCTIVE operation and requires user confirmation.
    Example: edit_file("C:/Proyectos/mi-app/main.py", "def old_name():", "def new_name():")
    """
    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not validated.exists() or not validated.is_file():
        return f"Error: File does not exist: {validated}"

    from security.validators import get_max_file_size_bytes
    max_bytes = get_max_file_size_bytes()
    try:
        if validated.stat().st_size > max_bytes:
            return f"Error: File too large to edit safely ({validated.stat().st_size:,} bytes)."
        original = validated.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

    occurrences = original.count(find)
    if occurrences == 0:
        return (
            "Error: no se encontró el texto indicado en el archivo. "
            "Vuelve a leer el archivo con read_file y copia el fragmento exacto."
        )

    replacements = occurrences if count == -1 else min(count, occurrences)
    new_content = original.replace(find, replace, replacements) if count != -1 else original.replace(find, replace)

    action_key = f"edit:{validated}:{hash(find)}:{hash(replace)}"
    if not is_confirmed(action_key):
        register_pending(
            key=action_key,
            description=(
                f"Editar '{validated.name}': reemplazar "
                f"{replacements} ocurrencia(s) de:\n  «{find[:120]}»\npor:\n  «{replace[:120]}»"
            ),
            action_type="edit_file",
            params={"path": str(validated), "find": find[:300], "replace": replace[:300]},
        )
        return CONFIRMATION_SENTINEL

    try:
        validated.write_text(new_content, encoding="utf-8")
        return f"Archivo editado correctamente: {validated} ({replacements} reemplazo(s))"
    except PermissionError:
        return f"Error: Permission denied writing to: {validated}"
    except Exception as exc:
        return f"Error writing file: {exc}"


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
