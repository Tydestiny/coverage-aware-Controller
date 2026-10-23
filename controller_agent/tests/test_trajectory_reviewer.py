"""Offline trajectory reviewer tests."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from controller_agent.results.reviewer import load_review_data, write_review_page
from scripts import experiment


def test_local_review_server_catalog_and_raw_log_access(run_dir: Path) -> None:
    """The browser can discover runs and open raw logs without file URLs."""
    from controller_agent.results.review_server import create_review_server

    server = create_review_server(run_dir.parents[3], port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/runs") as response:
            catalog = json.load(response)
        assert len(catalog["runs"]) == 1
        run_id = catalog["runs"][0]["id"]
        assert catalog["runs"][0]["experiment"] == "experiment"
        with urlopen(base + "/api/run?run=" + quote(run_id)) as response:
            data = json.load(response)
        assert len(data["rows"]) == 2
        assert data["raw_links"]["logger.log"].startswith("/raw?")
        with urlopen(base + data["raw_links"]["logger.log"]) as response:
            assert response.read() == b"original log"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/raw?run=" + quote(run_id) + "&name=../../manifest.json")
        assert error.value.code == 404
        with pytest.raises(HTTPError):
            urlopen(base + "/api/run?run=../../outside")
        with urlopen(base + "/") as response:
            html = response.read().decode("utf-8")
        assert 'id="experiment-select"' in html
        assert '__REVIEW_DATA__' not in html
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_review_annotations_save_reload_and_preserve_logs(run_dir: Path) -> None:
    """Per-run labels survive requests without modifying trajectory evidence."""
    from controller_agent.results.review_server import create_review_server

    original = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    server = create_review_server(run_dir.parents[3], port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    run_id = "experiment/tasks/task_1/controller"
    url = base + "/api/annotations?run=" + quote(run_id)
    document = {"task_id": "task_1", "branch": "controller", "run_path": str(run_dir.resolve()),
                "annotations": {"controller_1": {"timing": "late", "decision": "incorrect",
                                                  "guidance": "harmful", "note": "Observed failure."}},
                "step_annotations": {"agent_1": {"first_error": True, "note": "First unsupported action."}}}
    try:
        with urlopen(url) as response:
            assert json.load(response) is None
        request = Request(url, data=json.dumps(document).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request) as response:
            assert json.load(response)["annotations"] == document["annotations"]
        with urlopen(url) as response:
            assert json.load(response)["annotations"] == document["annotations"]
        with urlopen(url) as response:
            assert json.load(response)["step_annotations"] == document["step_annotations"]
        saved = run_dir / "review_annotations.json"
        assert json.loads(saved.read_text(encoding="utf-8"))["annotations"] == document["annotations"]
        assert {path: path.read_bytes() for path in original} == original
        bad = {**document, "annotations": {"not_a_call": {"timing": "late"}}}
        request = Request(url, data=json.dumps(bad).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 422
        assert json.loads(saved.read_text(encoding="utf-8"))["annotations"] == document["annotations"]
        bad = {**document, "step_annotations": {"unknown_agent_call": {"first_error": True}}}
        request = Request(url, data=json.dumps(bad).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 422
        request = Request(base + "/api/annotations?run=../../outside", data=json.dumps(document).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_browser_annotation_autosave_and_reload(run_dir: Path) -> None:
    """The local viewer persists a label and restores it after a page reload."""
    playwright = pytest.importorskip("playwright.sync_api")
    from controller_agent.results.review_server import create_review_server

    server = create_review_server(run_dir.parents[3], port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright.sync_playwright() as runtime:
            try:
                browser = runtime.chromium.launch(channel="chrome", headless=True)
            except Exception as exc:
                pytest.skip(f"Chrome is unavailable: {exc}")
            try:
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.locator("#detail .annotation-grid select").first.select_option("late")
                page.locator("#save-status").get_by_text("已保存").wait_for()
                page.locator("#detail .step-annotation input[type=checkbox]").check()
                page.locator("#save-status").get_by_text("已保存").wait_for()
                page.reload()
                assert page.locator("#detail .annotation-grid select").first.input_value() == "late"
                assert page.locator("#detail .step-annotation input[type=checkbox]").is_checked()
                assert json.loads((run_dir / "review_annotations.json").read_text(encoding="utf-8"))["annotations"]["controller_1"]["timing"] == "late"
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small synthetic JSONL fixture."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Create a run with one blocked and one approved action."""
    run = tmp_path / "artifacts" / "experiment" / "tasks" / "task_1" / "controller"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"task_id": "task_1", "branch": "controller", "split": "dev"}), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"termination_reason": "task_completed"}), encoding="utf-8")
    (run / "evaluator_result.json").write_text(json.dumps({"success": True}), encoding="utf-8")
    _jsonl(run / "continuation.jsonl", [
        {"step_number": 8, "agent_call_id": "agent_1", "agent_output": "</script><script>alert(1)</script>",
         "generated_code": "complete_task()", "controller_call_id": "controller_1", "execution_allowed": False,
         "environment_observation": None, "task_completed": False},
        {"step_number": 8, "agent_call_id": "agent_2", "agent_output": "verify()", "generated_code": "verify()",
         "controller_call_id": None, "execution_allowed": True, "environment_observation": ["ok"], "task_completed": False},
    ])
    _jsonl(run / "controller_calls.jsonl", [
        {"controller_call_id": "controller_1", "step_number": 8, "action": "REPLAN",
         "reason": "check target", "guidance": "verify first", "controller_input": {
             "task": "Do task", "agent_context_messages": [{"role": "user", "content": "Do task"}],
             "context_complete": True}, "controller_response": {"decision": {"action": "REPLAN"}}},
    ])
    _jsonl(run / "trigger_events.jsonl", [{"step_index": 8, "trigger_positive": True, "trigger_reason": "before_finish"}])
    _jsonl(run / "token_usage.jsonl", [
        {"call_id": "agent_1", "total_tokens": 10},
        {"call_id": "agent_2", "total_tokens": 12},
        {"call_id": "controller_1", "total_tokens": 5},
    ])
    raw = run / "raw_logs"
    raw.mkdir()
    (raw / "logger.log").write_text("original log", encoding="utf-8")
    return run


def test_load_review_data_joins_calls_and_repeated_step(run_dir: Path) -> None:
    """Call IDs distinguish two Agent generations sharing one step."""
    data = load_review_data(run_dir)
    assert len(data["rows"]) == 2
    first, second = data["rows"]
    assert first["controller_call"]["controller_input"]["agent_context_messages"][0]["content"] == "Do task"
    assert first["controller_tokens"]["total_tokens"] == 5
    assert first["trigger_events"][0]["trigger_positive"] is True
    assert second["controller_call"] is None
    assert second["agent_tokens"]["total_tokens"] == 12
    assert data["evaluator"]["success"] is True
    assert data["raw_links"]["logger.log"].startswith("file:")


def test_html_escapes_embedded_traces_and_keeps_sources(run_dir: Path, tmp_path: Path) -> None:
    """Untrusted Agent text cannot terminate the embedded JSON script."""
    original = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    output = tmp_path / "review.html"
    assert write_review_page(run_dir, output) == output.resolve()
    html = output.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)" in html
    assert "Agent 可见上下文" in html
    assert "导出标注 JSON" in html
    assert {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == original


def test_malformed_jsonl_and_mismatched_step_fail(run_dir: Path) -> None:
    """Bad evidence is surfaced instead of silently omitted."""
    calls = run_dir / "controller_calls.jsonl"
    calls.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="controller_calls.jsonl:1"):
        load_review_data(run_dir)
    _jsonl(calls, [{"controller_call_id": "controller_1", "step_number": 9}])
    with pytest.raises(ValueError, match="step_number differs"):
        load_review_data(run_dir)


def test_missing_optional_logs_are_reported(run_dir: Path) -> None:
    """Older runs remain readable but missing evidence is visible."""
    (run_dir / "trigger_events.jsonl").unlink()
    (run_dir / "token_usage.jsonl").unlink()
    data = load_review_data(run_dir)
    assert len(data["warnings"]) == 2
    assert data["rows"][0]["agent_tokens"] is None


def test_baseline_with_empty_controller_log_is_readable(run_dir: Path) -> None:
    """A baseline's empty Controller log is not treated as missing evidence."""
    (run_dir / "controller_calls.jsonl").write_text("", encoding="utf-8")
    _jsonl(run_dir / "continuation.jsonl", [
        {"step_number": 8, "agent_call_id": "agent_1", "agent_output": "verify()",
         "controller_call_id": None, "execution_allowed": True, "environment_observation": ["ok"]},
    ])
    data = load_review_data(run_dir)
    assert data["rows"][0]["controller_call"] is None
    assert not data["warnings"]


def test_baseline_step_annotation_needs_no_controller_call(run_dir: Path) -> None:
    """A baseline can record the first visible decision error independently."""
    from controller_agent.results.review_server import save_annotations

    (run_dir / "controller_calls.jsonl").write_text("", encoding="utf-8")
    document = {"task_id": "task_1", "branch": "controller", "run_path": str(run_dir.resolve()),
                "annotations": {},
                "step_annotations": {"agent_1": {"first_error": True, "note": "Authentication failed; retry was possible."}}}
    saved = save_annotations(run_dir, document)
    assert saved["annotations"] == {}
    assert saved["step_annotations"] == document["step_annotations"]
    with pytest.raises(ValueError, match="Only one step"):
        save_annotations(run_dir, {**document, "step_annotations": {
            "agent_1": {"first_error": True}, "agent_2": {"first_error": True}}})


def test_cli_review_writes_outside_source_run(run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The common CLI generates a local page without launching a model."""
    monkeypatch.setenv("CONTROLLER_ARTIFACT_ROOT", str(run_dir.parents[3]))
    monkeypatch.setenv("APPWORLD_ROOT", str(run_dir.parents[3] / "appworld"))
    assert experiment.main(["review", "--experiment-id", "experiment", "--task-id", "task_1", "--branch", "controller"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert Path(result["review_page"]).is_file()
    assert Path(result["review_page"]).parent.name == "reviews"
    with pytest.raises(ValueError, match="must not be written"):
        experiment.main(["review", "--experiment-id", "experiment", "--task-id", "task_1",
                         "--branch", "controller", "--output", str(run_dir / "review.html")])
