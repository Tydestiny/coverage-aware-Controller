"""Centralised configuration for all trigger detectors.

This module is the single source of truth for every tunable knob that the
trigger detectors (and their integration tests) consume.  Per-trigger dataclasses
let callers override only the fields they care about while keeping explicit
defaults for the current rule version.

Loading strategy:
- The default values declared here are the authoritative defaults.
- `load_trigger_config()` returns a fresh `TriggerConfig` snapshot.
- Callers may pass a `TriggerConfig` (or per-trigger dataclass overrides) to
  the detectors at construction time.  This is the recommended path so that
  tests/experiments can mutate behaviour without editing trigger source files.

Removed triggers:
- ``AfterErrorTrigger`` has been retired; its references in this file have
  been removed accordingly.  See ``triggers/README.md`` for the current
  trigger inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ── BeforeFinishTrigger ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BeforeFinishConfig:
    """Settings for the ``BeforeFinishTrigger``."""

    # Regex source (re-built at trigger construction time).  Default matches
    # ``apis.supervisor.complete_task(...)`` and ``supervisor.complete_task(...)``.
    pattern: str = r"\b(?:apis\.)?supervisor\.complete_task\s*\("


# ── BeforeWriteTrigger ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BeforeWriteConfig:
    """Settings for the ``BeforeWriteTrigger``."""

    # Keywords that mark a high-risk write/mutating API call.  Matched
    # case-insensitively against dot/underscore-delimited tokens of the
    # matched identifier.
    high_risk_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "delete", "update", "write", "create", "send", "pay",
            "purchase", "follow", "unfollow", "cancel", "remove",
            "modify", "set", "add", "block", "unblock",
            "accept", "apply", "approve", "assign", "attach", "clear",
            "compress", "copy", "decompress", "deny", "download", "exit",
            "forward", "initiate", "label", "like", "loop", "mark", "move",
            "next", "pause", "place", "play", "post", "previous", "record",
            "regenerate", "remind", "reply", "review", "seek", "settle",
            "shuffle", "snooze", "subscribe", "undelete", "unlabel", "unlike",
            "unsnooze", "upload", "withdraw",
        })
    )

    # Cap on the params preview stored in trigger metadata.
    params_preview_limit: int = 200

    # Public GET endpoints whose names contain a mutation keyword. Receipt
    # downloads return data; spotify.download_song changes library state.
    read_only_api_names: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "amazon.download_order_receipt",
            "amazon.download_prime_subscription_receipt",
            "amazon.show_last_product_purchase",
            "gmail.download_attachment",
            "splitwise.download_expense_receipt_file",
            "splitwise.download_payment_receipt_file",
            "spotify.show_song_review", "spotify.show_album_review",
            "spotify.show_playlist_review",
            "spotify.download_premium_subscription_receipt",
            "todoist.show_label", "todoist.download_attachment",
            "venmo.download_bank_transfer_receipt",
            "venmo.download_transaction_receipt",
        })
    )


# ── NoProgressTrigger ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoProgressConfig:
    """Settings for the ``NoProgressTrigger``."""

    # How many recent steps to look back in history.
    lookback_window: int = 5

    # repeated_action: same logical action appears >= this many times in the
    # lookback window (including current step).
    repeated_action_threshold: int = 3

    # Skip counting / matching ``apis.api_docs.*`` calls.  These are normal
    # exploration and would otherwise create false positives.
    exclude_api_docs: bool = True

    # Regex fragment used to detect api_docs calls when ``exclude_api_docs``
    # is enabled.  Case-insensitive.
    api_docs_pattern: str = r"apis\.api_docs\."

    # Suffix tokens that mark an identifier as an error/exception name.
    error_suffixes: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "error", "exception", "failed", "failure",
            "denied", "forbidden", "unauthorized",
            "notfound", "not_found",
            "conflict", "timeout",
        })
    )

    # Generic error keywords that, when they appear as whole words, lower
    # the fingerprint priority to ``generic_error``.
    generic_error_keywords: Sequence[str] = field(
        default_factory=lambda: ("error", "failed", "failure")
    )

    # Minimum fingerprint length to count as a real error (filters out
    # placeholders like ``exc:str``).
    min_fingerprint_length: int = 8


# ── Composite / budget ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TriggerBudgetConfig:
    """Settings for the composite trigger dispatcher and per-task budget."""

    # Trigger detectors registered with the composite dispatcher, in priority
    # order.  First match wins.
    detector_order: tuple[str, ...] = (
        "before_finish",
        "before_write",
        "no_progress",
    )

    # Maximum number of trigger evaluations that may invoke the Controller
    # within a single task.  After this, the composite returns "budget_exhausted"
    # unless the ``min_before_finish`` floor is not yet met.
    max_triggers_per_task: int = 4

    # Minimum number of before_finish triggers guaranteed before the budget
    # cap kicks in.  Ensures the task completion is always inspected.
    min_before_finish_triggers: int = 1


# ── Controller / runner budget (used by integration tests) ────────────────────


@dataclass(frozen=True)
class ControllerBudgetConfig:
    """Settings for the Controller and the ControlledAppWorldRunner budget."""

    # Inside ``ControlledReActAgent``.
    max_controller_calls_per_task: int = 5
    max_controller_blocks: int = 5
    max_consecutive_blocks: int = 3

    # ``RunnerBudget`` parameters that mirror the controller limits.
    max_controller_calls: int = 5
    max_controller_blocks_runner: int = 5
    max_extra_agent_turns_after_block: int = 3


# ── Top-level ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TriggerConfig:
    """Single source of truth for every trigger tunable."""

    before_finish: BeforeFinishConfig = field(default_factory=BeforeFinishConfig)
    before_write: BeforeWriteConfig = field(default_factory=BeforeWriteConfig)
    no_progress: NoProgressConfig = field(default_factory=NoProgressConfig)
    budget: TriggerBudgetConfig = field(default_factory=TriggerBudgetConfig)
    controller: ControllerBudgetConfig = field(default_factory=ControllerBudgetConfig)


# Singleton default instance — used by detectors that do not receive an
# explicit config so that the historical defaults are preserved.
DEFAULT_CONFIG: TriggerConfig = TriggerConfig()


def load_trigger_config() -> TriggerConfig:
    """Return a fresh snapshot of the default :class:`TriggerConfig`.

    Currently the config is in-process, but exposing a function keeps the
    call-site stable if we later load overrides from disk or env vars.
    """
    return TriggerConfig(
        before_finish=BeforeFinishConfig(),
        before_write=BeforeWriteConfig(),
        no_progress=NoProgressConfig(),
        budget=TriggerBudgetConfig(),
        controller=ControllerBudgetConfig(),
    )
