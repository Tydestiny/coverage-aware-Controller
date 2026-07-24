"""Deterministic Controller policy distilled from Part3 branch experiments.

Implements Prompt G0 priors without requiring a live LLM call.
Can also serve as a safe fallback / dry-run backend.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from controller_agent.state.schemas import (
    Action,
    ControllerDecision,
    ControllerInput,
    ControllerResponse,
)


_PROMPTS = Path(__file__).resolve().parent / "prompts"
_TEMPLATES = json.loads((_PROMPTS / "instruction_templates.json").read_text(encoding="utf-8"))

_WRITE_RE = re.compile(
    r"\b(create_|add_|remove_|delete_|move_|update_|send_|pay_|subscribe_|upload_|"
    r"create_transaction|create_playlist|add_song|remove_song|move_file|delete_file|"
    r"create_directory|compress_|request_money)\w*\s*\(",
    re.I,
)
_COMPLETE_RE = re.compile(r"\b(?:apis\.)?supervisor\.complete_task\s*\(", re.I)
_EXCUSE_RE = re.compile(
    r"complete_task\s*\(\s*answer\s*=\s*['\"].{0,200}(cannot|unable|fail|partial|"
    r"insufficient|restriction|not possible|no songs need|as a proxy|instead)",
    re.I,
)
_SET_EVIDENCE_RE = re.compile(
    r"(remove_set\s*=|songs_to_remove|albums_to_remove|to_remove\s*=|"
    r"neither liked|not liked and not downloaded|set difference|"
    r"library\s*-\s*liked|candidate(_ids|_list)?\s*=)",
    re.I,
)
_MEMBERSHIP_CLEANUP_RE = re.compile(
    r"(liked and downloaded|neither liked|not liked|not downloaded|"
    r"cleanup.*(library|song|album)|keep only.*(liked|downloaded))",
    re.I,
)
_API_UNKNOWN_RE = re.compile(
    r"(unexpected keyword|not authorized|401|422|release_year|show_api_doc|"
    r"access_token|NameError|TypeError)",
    re.I,
)
_GOAL_SUB_RE = re.compile(r"(as a proxy|instead of|workaround|liked songs as)", re.I)
_SUCCESS_RE = re.compile(
    r"(Sent money|Created playlist|successfully|Moved |transaction_id)",
    re.I,
)
_ATTR_FILTER_RE = re.compile(
    r"(release_year|release year|before \d{4}|after \d{4}|created_at|"
    r"older than|newer than|year\b)",
    re.I,
)


def infer_case_type(controller_input: ControllerInput) -> str:
    """Map trigger + proposed code to offline case_type labels."""
    reason = (controller_input.trigger_reason or "").lower()
    code = controller_input.current_agent_output or ""
    if "before_finish" in reason or "complete" in reason or _COMPLETE_RE.search(code):
        return "first_complete_task_before"
    if "write" in reason or "high_risk" in reason or _WRITE_RE.search(code):
        return "first_high_risk_write_before"
    if "error" in reason or controller_input.recent_errors:
        return "second_same_error_after"
    return "unknown"


def _hist(controller_input: ControllerInput) -> str:
    # Prior context only — exclude current_agent_output so proposed comments /
    # variable names cannot fake set-evidence.
    parts = [
        "\n".join(controller_input.recent_errors or []),
        "\n".join(controller_input.completed_operations or []),
        json.dumps(controller_input.recent_history[-6:], ensure_ascii=False)[:4000],
    ]
    return "\n".join(parts)


def _instr(action: str, slot: str) -> str:
    templates = json.loads((_PROMPTS / "instruction_templates.json").read_text(encoding="utf-8"))
    block = templates.get(action, {})
    return str(block.get(slot) or block.get("generic") or "")


class PolicyG0Backend:
    """Rule backend implementing G0 priors from branch experiments."""

    def __call__(self, controller_input: ControllerInput) -> ControllerResponse:
        decision = self.decide(controller_input)
        return ControllerResponse(decision=decision, model_name="policy_g0")

    def decide(self, controller_input: ControllerInput) -> ControllerDecision:
        case_type = infer_case_type(controller_input)
        code = controller_input.current_agent_output or ""
        hist = _hist(controller_input)
        errors = "\n".join(controller_input.recent_errors or [])

        if case_type == "first_complete_task_before":
            return self._complete(code, hist, errors, controller_input.task or "")
        if case_type == "first_high_risk_write_before":
            return self._write(code, hist, errors, controller_input.task or "")
        if case_type == "second_same_error_after":
            return self._error(code, hist, errors)
        return ControllerDecision(Action.PROCEED, "no_case_type_match", "", 0.4)

    def _complete(self, code: str, hist: str, errors: str, task: str = "") -> ControllerDecision:
        blob = hist + "\n" + code
        task_blob = f"{task}\n{blob}"

        if _EXCUSE_RE.search(code) or _GOAL_SUB_RE.search(blob):
            return ControllerDecision(
                Action.BLOCK_RISKY_ACTION,
                "premature_complete_with_excuse_or_goal_sub",
                _instr("BLOCK_RISKY_ACTION", "complete_before"),
                0.9,
            )

        # Year / attribute-filter cleanups: vague REPLAN hurt online (ce359b5_1).
        # Prefer VERIFY with an attribute checklist; LOOKUP only if detail API unseen.
        if _ATTR_FILTER_RE.search(task_blob):
            has_attr_fetch = bool(
                re.search(r"show_song\(|release_year|release_date|created_at", hist, re.I)
            )
            if (not has_attr_fetch) and (not re.search(r"show_api_doc", hist, re.I)):
                return ControllerDecision(
                    Action.LOOKUP_API_SPEC,
                    "complete_attribute_filter_needs_detail_api",
                    _instr("LOOKUP_API_SPEC", "attribute_filter_complete"),
                    0.75,
                )
            return ControllerDecision(
                Action.VERIFY_TARGET_SET,
                "complete_attribute_filter_needs_printed_table",
                _instr("VERIFY_TARGET_SET", "attribute_filter_complete"),
                0.8,
            )

        if re.search(r"liked|download|library|album|playlist|membership|remove", blob, re.I) and not _SET_EVIDENCE_RE.search(
            hist
        ):
            return ControllerDecision(
                Action.VERIFY_TARGET_SET,
                "complete_without_membership_printout",
                _instr("VERIFY_TARGET_SET", "complete_before"),
                0.75,
            )
        if errors and not _SUCCESS_RE.search(hist):
            return ControllerDecision(
                Action.BLOCK_RISKY_ACTION,
                "complete_while_errors_unresolved",
                _instr("BLOCK_RISKY_ACTION", "complete_before"),
                0.8,
            )
        return ControllerDecision(
            Action.REPLAN,
            "complete_before_default_replan",
            _instr("REPLAN", "complete_before"),
            0.65,
        )

    def _write(self, code: str, hist: str, errors: str, task: str = "") -> ControllerDecision:
        blob = f"{task}\n{hist}\n{code}"
        if _API_UNKNOWN_RE.search(errors) or (
            re.search(r"release_year|created_at", code + hist, re.I)
            and not re.search(r"show_api_doc|show_song\(|created_at\s*=", hist, re.I)
        ):
            if re.search(r"release_year|unexpected keyword|401|422|not authorized", errors + code, re.I):
                return ControllerDecision(
                    Action.LOOKUP_API_SPEC,
                    "write_needs_api_or_field_lookup",
                    _instr("LOOKUP_API_SPEC", "write_before"),
                    0.7,
                )

        # Evidence must appear in prior history, not in the proposed write's comments
        # / variable names (e.g. "# Remove songs not liked..." falsely matched).
        has_set_evidence = bool(_SET_EVIDENCE_RE.search(hist))
        about_to_write = bool(_WRITE_RE.search(code))
        membership = bool(_MEMBERSHIP_CLEANUP_RE.search(blob))

        # Library liked/downloaded cleanup: soft VERIFY with explicit remove_set
        # formula. Hard BLOCK over-corrected online when CONTINUE already strong.
        if membership and about_to_write:
            reason = (
                "membership_cleanup_write_verify_remove_set"
                if has_set_evidence
                else "membership_cleanup_write_needs_remove_set"
            )
            return ControllerDecision(
                Action.VERIFY_TARGET_SET,
                reason,
                _instr("VERIFY_TARGET_SET", "membership_cleanup_write"),
                0.8 if not has_set_evidence else 0.7,
            )

        if about_to_write and not has_set_evidence:
            return ControllerDecision(
                Action.BLOCK_RISKY_ACTION,
                "irreversible_write_without_verified_set",
                _instr("BLOCK_RISKY_ACTION", "write_before"),
                0.85,
            )

        if about_to_write:
            return ControllerDecision(
                Action.VERIFY_TARGET_SET,
                "write_before_default_verify",
                _instr("VERIFY_TARGET_SET", "write_before"),
                0.7,
            )

        return ControllerDecision(Action.PROCEED, "write_trigger_but_no_write_code", "", 0.5)

    def _error(self, code: str, hist: str, errors: str) -> ControllerDecision:
        if re.search(r"show_api_doc|access_token\s*=", code, re.I):
            return ControllerDecision(Action.PROCEED, "already_repairing", "", 0.8)
        if _GOAL_SUB_RE.search(code + "\n" + hist):
            return ControllerDecision(
                Action.REPLAN,
                "error_followed_by_goal_substitution",
                _instr("REPLAN", "error_after"),
                0.85,
            )
        if re.search(r"unexpected keyword|not authorized|401|422", errors, re.I):
            return ControllerDecision(
                Action.LOOKUP_API_SPEC,
                "api_param_or_auth_error",
                _instr("LOOKUP_API_SPEC", "error_after"),
                0.8,
            )
        return ControllerDecision(
            Action.REPLAN,
            "generic_error_recovery",
            _instr("REPLAN", "error_after"),
            0.6,
        )


def load_g0_system_prompt() -> str:
    return (_PROMPTS / "g0_system.txt").read_text(encoding="utf-8")


def render_g0_user_prompt(controller_input: ControllerInput) -> str:
    template = (_PROMPTS / "g0_user_template.txt").read_text(encoding="utf-8")
    hist = json.dumps(controller_input.recent_history[-8:], ensure_ascii=False)
    if len(hist) > 6000:
        hist = hist[:6000] + "\n...[truncated]"
    return (
        template.replace("{{trigger_reason}}", controller_input.trigger_reason or "")
        .replace("{{case_type_hint}}", infer_case_type(controller_input))
        .replace("{{task}}", controller_input.task or "")
        .replace("{{current_agent_output}}", controller_input.current_agent_output or "")
        .replace("{{recent_errors}}", "\n".join(controller_input.recent_errors or []) or "(none)")
        .replace(
            "{{completed_operations}}",
            "\n".join(controller_input.completed_operations or []) or "(none)",
        )
        .replace("{{known_entities}}", ", ".join(controller_input.known_entities or []) or "(none)")
        .replace("{{recent_history}}", hist or "(none)")
    )
