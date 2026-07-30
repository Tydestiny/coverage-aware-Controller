"""Deterministic Prompt-G0 policy and renderer."""

from __future__ import annotations

import json
import re
from pathlib import Path

from controller_agent.state.schemas import Action, ControllerDecision, ControllerInput, ControllerResponse


_PROMPTS = Path(__file__).resolve().parent / "prompts"
_WRITE_RE = re.compile(r"\b(create_|add_|remove_|delete_|move_|update_|send_|pay_|subscribe_|upload_)\w*\s*\(", re.I)
_COMPLETE_RE = re.compile(r"\b(?:apis\.)?supervisor\.complete_task\s*\(", re.I)
_EXCUSE_RE = re.compile(r"complete_task\s*\(\s*answer\s*=\s*['\"].{0,200}(cannot|unable|fail|partial|insufficient|restriction|not possible|as a proxy|instead)", re.I)
_SET_EVIDENCE_RE = re.compile(r"(remove_set\s*=|songs_to_remove|albums_to_remove|to_remove\s*=|set difference|library\s*-\s*liked)", re.I)
_MEMBERSHIP_RE = re.compile(r"(liked and downloaded|neither liked|not liked|not downloaded|cleanup.*(library|song|album)|keep only.*(liked|downloaded))", re.I)
_API_UNKNOWN_RE = re.compile(r"(unexpected keyword|not authorized|401|422|release_year|show_api_doc|access_token|NameError|TypeError)", re.I)
_GOAL_SUB_RE = re.compile(r"(as a proxy|instead of|workaround|liked songs as)", re.I)
_SUCCESS_RE = re.compile(r"(Sent money|Created playlist|successfully|Moved |transaction_id)", re.I)
_ATTR_FILTER_RE = re.compile(r"(release_year|release year|before \d{4}|after \d{4}|created_at|older than|newer than|year\b)", re.I)


def _hist(controller_input: ControllerInput) -> str:
    """Render the complete, exact context passed to the Agent model."""
    return json.dumps(controller_input.agent_context_messages, ensure_ascii=False)


def infer_case_type(controller_input: ControllerInput) -> str:
    """Map the trigger and proposed output to a deterministic case type."""
    reason = controller_input.trigger_reason.lower()
    output = controller_input.proposed_agent_output
    if "before_finish" in reason or "complete" in reason or _COMPLETE_RE.search(output):
        return "first_complete_task_before"
    if "write" in reason or "high_risk" in reason or _WRITE_RE.search(output):
        return "first_high_risk_write_before"
    if "error" in reason or re.search(r"(?:traceback|exception|error)\b", _hist(controller_input), re.I):
        return "second_same_error_after"
    return "unknown"


def _instruction(action: str, slot: str) -> str:
    """Read one fixed instruction template without changing policy inputs."""
    templates = json.loads((_PROMPTS / "instruction_templates.json").read_text(encoding="utf-8"))
    block = templates.get(action, {})
    return str(block.get(slot) or block.get("generic") or "")


class PolicyG0Backend:
    """Rule backend implementing the existing G0 action taxonomy."""

    def __call__(self, controller_input: ControllerInput) -> ControllerResponse:
        """Return a response compatible with the live Controller backend."""
        return ControllerResponse(decision=self.decide(controller_input), model_name="policy_g0")

    def decide(self, controller_input: ControllerInput) -> ControllerDecision:
        """Choose a conservative action from complete causal context."""
        case_type = infer_case_type(controller_input)
        output = controller_input.proposed_agent_output
        history = _hist(controller_input)
        if case_type == "first_complete_task_before":
            return self._complete(output, history, controller_input.task)
        if case_type == "first_high_risk_write_before":
            return self._write(output, history, controller_input.task)
        if case_type == "second_same_error_after":
            return self._error(output, history)
        return ControllerDecision(Action.PROCEED, "no_case_type_match")

    def _complete(self, output: str, history: str, task: str) -> ControllerDecision:
        blob = f"{history}\n{output}"
        if _EXCUSE_RE.search(output) or _GOAL_SUB_RE.search(blob):
            return ControllerDecision(Action.REPLAN, "premature_complete_with_excuse_or_goal_sub", _instruction("REPLAN", "complete_before"))
        if _ATTR_FILTER_RE.search(f"{task}\n{blob}"):
            if not re.search(r"show_song\(|release_year|release_date|created_at|show_api_doc", history, re.I):
                return ControllerDecision(Action.LOOKUP_API_SPEC, "complete_attribute_filter_needs_detail_api", _instruction("LOOKUP_API_SPEC", "attribute_filter_complete"))
            return ControllerDecision(Action.VERIFY_TARGET_SET, "complete_attribute_filter_needs_printed_table", _instruction("VERIFY_TARGET_SET", "attribute_filter_complete"))
        if _MEMBERSHIP_RE.search(blob) and not _SET_EVIDENCE_RE.search(history):
            return ControllerDecision(Action.VERIFY_TARGET_SET, "complete_without_membership_printout", _instruction("VERIFY_TARGET_SET", "complete_before"))
        if re.search(r"(?:traceback|exception|error)\b", history, re.I) and not _SUCCESS_RE.search(history):
            return ControllerDecision(Action.REPLAN, "complete_while_errors_unresolved", _instruction("REPLAN", "complete_before"))
        return ControllerDecision(Action.REPLAN, "complete_before_default_replan", _instruction("REPLAN", "complete_before"))

    def _write(self, output: str, history: str, task: str) -> ControllerDecision:
        blob = f"{task}\n{history}\n{output}"
        if _API_UNKNOWN_RE.search(history) and re.search(r"release_year|unexpected keyword|401|422|not authorized", history + output, re.I):
            return ControllerDecision(Action.LOOKUP_API_SPEC, "write_needs_api_or_field_lookup", _instruction("LOOKUP_API_SPEC", "write_before"))
        if _MEMBERSHIP_RE.search(blob) and _WRITE_RE.search(output):
            reason = "membership_cleanup_write_verify_remove_set" if _SET_EVIDENCE_RE.search(history) else "membership_cleanup_write_needs_remove_set"
            return ControllerDecision(Action.VERIFY_TARGET_SET, reason, _instruction("VERIFY_TARGET_SET", "membership_cleanup_write"))
        if _WRITE_RE.search(output) and not _SET_EVIDENCE_RE.search(history):
            return ControllerDecision(Action.REPLAN, "irreversible_write_without_verified_set", _instruction("REPLAN", "write_before"))
        if _WRITE_RE.search(output):
            return ControllerDecision(Action.VERIFY_TARGET_SET, "write_before_default_verify", _instruction("VERIFY_TARGET_SET", "write_before"))
        return ControllerDecision(Action.PROCEED, "write_trigger_but_no_write_code")

    def _error(self, output: str, history: str) -> ControllerDecision:
        if re.search(r"show_api_doc|access_token\s*=", output, re.I):
            return ControllerDecision(Action.PROCEED, "already_repairing")
        if _GOAL_SUB_RE.search(f"{output}\n{history}"):
            return ControllerDecision(Action.REPLAN, "error_followed_by_goal_substitution", _instruction("REPLAN", "error_after"))
        if re.search(r"unexpected keyword|not authorized|401|422", history, re.I):
            return ControllerDecision(Action.LOOKUP_API_SPEC, "api_param_or_auth_error", _instruction("LOOKUP_API_SPEC", "error_after"))
        return ControllerDecision(Action.REPLAN, "generic_error_recovery", _instruction("REPLAN", "error_after"))


def load_g0_system_prompt() -> str:
    """Load the fixed G0 system prompt."""
    return (_PROMPTS / "g0_system.txt").read_text(encoding="utf-8")


def render_g0_user_prompt(controller_input: ControllerInput) -> str:
    """Render complete ControllerInput without secondary truncation."""
    template = (_PROMPTS / "g0_user_template.txt").read_text(encoding="utf-8")
    return (
        template.replace("{{trigger_reason}}", controller_input.trigger_reason)
        .replace("{{trigger_metadata}}", json.dumps(controller_input.trigger_metadata, ensure_ascii=False))
        .replace("{{context_complete}}", str(controller_input.context_complete).lower())
        .replace("{{task}}", controller_input.task)
        .replace("{{proposed_agent_output}}", controller_input.proposed_agent_output)
        .replace("{{proposed_execution}}", controller_input.proposed_execution)
        .replace("{{previous_controller_decisions}}", json.dumps(controller_input.previous_controller_decisions, ensure_ascii=False))
        .replace("{{agent_context_messages}}", json.dumps(controller_input.agent_context_messages, ensure_ascii=False))
    )
