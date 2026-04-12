"""
security/validators.py

Path validation module for AMADEUS.
All filesystem tools must call validate_path() before any I/O operation.
This module has NO imports from agent/ or streamlit to avoid circular dependencies.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Directories that are never accessible, even if inside an allowed base dir
_BLOCKED_SYSTEM_DIRS = {
    # Windows
    "C:/Windows", "C:\\Windows",
    "C:/Windows/System32", "C:\\Windows\\System32",
    "C:/Program Files", "C:\\Program Files",
    "C:/Program Files (x86)", "C:\\Program Files (x86)",
    # Unix
    "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/boot", "/dev", "/proc", "/sys", "/root",
}


class PathSecurityError(PermissionError):
    """Raised when a path fails security validation."""
    pass


def get_allowed_dirs() -> list[Path]:
    """
    Read ALLOWED_BASE_DIRS from the environment (comma-separated paths).
    Returns resolved (absolute, symlink-free) Path objects.
    Falls back to the user's home directory if the env var is not set.
    """
    raw = os.getenv("ALLOWED_BASE_DIRS", "").strip()
    if not raw:
        return [Path.home().resolve()]

    dirs: list[Path] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if entry:
            try:
                resolved = Path(entry).expanduser().resolve()
                dirs.append(resolved)
            except Exception:
                pass  # Skip malformed entries silently

    return dirs if dirs else [Path.home().resolve()]


def get_max_file_size_bytes() -> int:
    """Return the maximum allowed file read size in bytes (from env var)."""
    try:
        mb = float(os.getenv("MAX_FILE_READ_MB", "10"))
    except ValueError:
        mb = 10.0
    return int(mb * 1024 * 1024)


def validate_path(raw_path: str) -> Path:
    """
    Validate and resolve a path string.

    Steps:
      1. Expand ~ and environment variables
      2. Resolve to absolute, symlink-free path
      3. Reject paths inside hardcoded system directories
      4. Confirm the path is inside at least one allowed base directory

    Returns the resolved Path if valid.
    Raises PathSecurityError if validation fails.
    """
    if not raw_path or not raw_path.strip():
        raise PathSecurityError("Path cannot be empty.")

    try:
        # expanduser handles ~ notation; resolve() makes absolute + follows symlinks
        resolved = Path(raw_path.strip()).expanduser().resolve()
    except Exception as exc:
        raise PathSecurityError(f"Cannot resolve path '{raw_path}': {exc}") from exc

    # Block hardcoded system directories
    resolved_str = resolved.as_posix()
    for blocked in _BLOCKED_SYSTEM_DIRS:
        blocked_path = Path(blocked).resolve()
        try:
            if resolved == blocked_path or resolved.is_relative_to(blocked_path):
                raise PathSecurityError(
                    f"Access denied: '{resolved}' is a protected system directory."
                )
        except ValueError:
            pass  # is_relative_to raises ValueError on different drives (Windows)

    # Check against allowed base directories
    allowed = get_allowed_dirs()
    for base in allowed:
        try:
            if resolved == base or resolved.is_relative_to(base):
                return resolved
        except ValueError:
            # Different drives on Windows - not a match
            continue

    raise PathSecurityError(
        f"Access denied: '{resolved}' is outside the allowed directories.\n"
        f"Allowed: {', '.join(str(d) for d in allowed)}\n"
        f"Set ALLOWED_BASE_DIRS in your .env file to expand access."
    )
