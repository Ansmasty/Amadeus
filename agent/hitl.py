"""
agent/hitl.py

Human-in-the-Loop (HITL) confirmation registry for AMADEUS.

Uses module-level state (safe for local single-user deployment where Ollama
is bound to 127.0.0.1). Tools check/register confirmations here; the Streamlit
UI in main.py polls and resolves them.

No imports from Streamlit or other agent modules to avoid circular dependencies.
"""
from dataclasses import dataclass, field
from typing import Any


# Tools return this sentinel string when they need user confirmation.
# The system prompt tells the LLM to stop and display the confirmation widget.
CONFIRMATION_SENTINEL = "__AMADEUS_NEEDS_CONFIRMATION__"

# ─── Data Model ───────────────────────────────────────────────────────────────


@dataclass
class PendingAction:
    """Represents a destructive action awaiting user confirmation."""
    key: str                        # Unique identifier, e.g. "delete:/home/user/foo.txt"
    description: str                # Short human-readable question for the user
    action_type: str                # e.g. "delete_file", "move_file", "copy_file"
    params: dict[str, Any] = field(default_factory=dict)  # Action parameters to display


# ─── Module-Level State ───────────────────────────────────────────────────────

_confirmed_keys: set[str] = set()
_pending: dict[str, PendingAction] = {}

# ─── Public API ───────────────────────────────────────────────────────────────


def is_confirmed(key: str) -> bool:
    """Return True if the user has already confirmed the action with this key."""
    return key in _confirmed_keys


def register_pending(
    key: str,
    description: str,
    action_type: str,
    params: dict[str, Any],
) -> None:
    """Register an action that requires user confirmation before proceeding."""
    _pending[key] = PendingAction(
        key=key,
        description=description,
        action_type=action_type,
        params=params,
    )


def get_pending_action() -> PendingAction | None:
    """Return the most recently registered pending action, or None if none exist."""
    if _pending:
        return next(iter(_pending.values()))
    return None


def confirm(key: str) -> None:
    """Mark a pending action as confirmed by the user."""
    _confirmed_keys.add(key)
    _pending.pop(key, None)


def cancel(key: str) -> None:
    """Cancel a pending action (user declined)."""
    _pending.pop(key, None)
    _confirmed_keys.discard(key)


def reset_all() -> None:
    """Clear all pending and confirmed state. Call after each completed agent turn."""
    _confirmed_keys.clear()
    _pending.clear()
