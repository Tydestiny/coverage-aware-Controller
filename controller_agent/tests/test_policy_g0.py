from controller_agent.controller.controller import Controller
from controller_agent.controller.policy_g0 import PolicyG0Backend, infer_case_type, load_g0_system_prompt
from controller_agent.state.schemas import Action, ControllerInput


def _inp(**kwargs) -> ControllerInput:
    base = dict(
        task="Cleanup library: keep liked or downloaded only.",
        trigger_reason="before_finish",
        recent_history=[],
        current_agent_output="apis.supervisor.complete_task(answer='cannot finish')",
        known_entities=[],
        completed_operations=[],
        recent_errors=[],
    )
    base.update(kwargs)
    return ControllerInput(**base)


def test_infer_complete_case_type() -> None:
    assert infer_case_type(_inp()) == "first_complete_task_before"


def test_block_excuse_complete() -> None:
    d = PolicyG0Backend().decide(_inp())
    assert d.action == Action.BLOCK_RISKY_ACTION
    assert "complete" in d.instruction.lower() or "Do not" in d.instruction


def test_write_without_set_gets_block() -> None:
    d = PolicyG0Backend().decide(
        _inp(
            trigger_reason="before_first_write",
            current_agent_output="apis.spotify.remove_song_from_library(song_id=1, access_token=t)",
            recent_history=[{"role": "assistant", "content": "I will remove songs now"}],
        )
    )
    assert d.action == Action.BLOCK_RISKY_ACTION


def test_controller_adapter_accepts_policy_g0() -> None:
    ctrl = Controller(backend=PolicyG0Backend())
    resp = ctrl.decide(_inp())
    assert resp.decision.action == Action.BLOCK_RISKY_ACTION
    assert resp.model_name == "policy_g0"


def test_g0_prompt_loads() -> None:
    text = load_g0_system_prompt()
    assert "VERIFY_TARGET_SET" in text
    assert "first_high_risk_write_before" in text
