"""Local trajectory review server and per-run audit annotations."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from controller_agent.results.reviewer import load_review_data

RAW_FILES = frozenset({"logger.log", "logger.jsonl", "lm_calls.jsonl", "api_calls.jsonl", "environment_io.md"})
ANNOTATION_FILE = "review_annotations.json"
ANNOTATION_FIELDS = {
    "timing": {"", "appropriate", "early", "late", "unclear"},
    "decision": {"", "supported", "unsupported", "incorrect", "unclear"},
    "guidance": {"", "actionable", "vague", "overreaching", "harmful", "not_applicable"},
}


def validate_annotations(value: object, directory: Path) -> dict:
    """Require annotations to match the selected run and its Controller calls."""
    if not isinstance(value, dict) or not isinstance(value.get("annotations"), dict):
        raise ValueError("Expected an annotations object")
    step_annotations = value.get("step_annotations", {})
    if not isinstance(step_annotations, dict):
        raise ValueError("Expected a step_annotations object")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (value.get("task_id") != manifest.get("task_id")
            or value.get("branch") != manifest.get("branch")
            or value.get("run_path") != str(directory)):
        raise ValueError("Annotation identity does not match the selected run")
    calls = set()
    call_file = directory / "controller_calls.jsonl"
    if call_file.is_file():
        for line in call_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                calls.add(json.loads(line)["controller_call_id"])
    for call_id, record in value["annotations"].items():
        if call_id not in calls or not isinstance(record, dict):
            raise ValueError(f"Unknown Controller call or invalid annotation: {call_id}")
        if any(key not in {*ANNOTATION_FIELDS, "note"} for key in record):
            raise ValueError(f"Unexpected annotation field for {call_id}")
        if any(not isinstance(record.get(key, ""), str) or record.get(key, "") not in choices
               for key, choices in ANNOTATION_FIELDS.items()):
            raise ValueError(f"Invalid audit label for {call_id}")
        if not isinstance(record.get("note", ""), str):
            raise ValueError(f"Audit note must be text for {call_id}")
    agent_calls = set()
    for line in (directory / "continuation.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            agent_calls.add(json.loads(line).get("agent_call_id"))
    first_errors = 0
    for call_id, record in step_annotations.items():
        if call_id not in agent_calls or not isinstance(record, dict):
            raise ValueError(f"Unknown Agent call or invalid step annotation: {call_id}")
        if set(record) - {"first_error", "note"}:
            raise ValueError(f"Unexpected step annotation field for {call_id}")
        if not isinstance(record.get("first_error", False), bool) or not isinstance(record.get("note", ""), str):
            raise ValueError(f"Invalid step annotation for {call_id}")
        first_errors += bool(record.get("first_error", False))
    if first_errors > 1:
        raise ValueError("Only one step can be marked as the first error")
    return {"task_id": manifest["task_id"], "branch": manifest["branch"],
            "run_path": str(directory), "annotations": value["annotations"],
            "step_annotations": step_annotations,
            "updated_at": datetime.now(timezone.utc).isoformat()}


def save_annotations(directory: Path, value: object) -> dict:
    """Atomically replace only this run's annotation file."""
    document = validate_annotations(value, directory)
    target = directory / ANNOTATION_FILE
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".review_annotations_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return document


def review_catalog(root: Path) -> dict:
    """List existing readable runs without loading trajectory contents."""
    root = root.resolve()
    runs, errors = [], []
    for manifest in sorted(root.glob("*/tasks/**/manifest.json")):
        directory = manifest.parent.resolve()
        if not directory.is_relative_to(root) or not (directory / "continuation.jsonl").is_file():
            continue
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(info, dict):
                raise ValueError("Manifest must be an object")
            relative = directory.relative_to(root)
            runs.append({"id": relative.as_posix(), "experiment": relative.parts[0],
                         "task_id": info.get("task_id") or directory.parent.name,
                         "branch": info.get("branch") or directory.name,
                         "checkpoint_id": info.get("checkpoint_id"), "split": info.get("split")})
        except (ValueError, OSError) as exc:
            errors.append(f"{manifest}: {exc}")
    return {"runs": runs, "errors": errors}


def create_review_server(root: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Bind locally and allow annotation writes only for cataloged runs."""
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: object, status: int = 200) -> None:
            self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def _run_directory(self, run_id: str | None) -> Path | None:
            catalog = review_catalog(root)
            if run_id not in {run["id"] for run in catalog["runs"]}:
                return None
            directory = (root / run_id).resolve()
            return directory if directory.is_relative_to(root) else None

        def do_GET(self) -> None:
            """Serve the viewer, run catalog, annotations and allowlisted raw logs."""
            request = urlsplit(self.path)
            params = parse_qs(request.query)
            try:
                if request.path == "/":
                    template = Path(__file__).with_name("reviewer.html").read_text(encoding="utf-8")
                    self._send(template.replace("__REVIEW_DATA__", "null").encode("utf-8"), "text/html; charset=utf-8")
                    return
                catalog = review_catalog(root)
                if request.path == "/api/runs":
                    self._json(catalog)
                    return
                run_id = params.get("run", [None])[0]
                directory = self._run_directory(run_id)
                if directory is None:
                    self._json({"error": "Run not found in the active experiment catalog"}, 404)
                    return
                if request.path == "/api/run":
                    data = load_review_data(directory)
                    data["raw_links"] = {name: f"/raw?run={quote(run_id, safe='')}&name={quote(name)}"
                                         for name in data["raw_links"]}
                    self._json(data)
                    return
                if request.path == "/api/annotations":
                    path = directory / ANNOTATION_FILE
                    self._json(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None)
                    return
                if request.path == "/raw":
                    name = params.get("name", [""])[0]
                    path = (directory / "raw_logs" / name).resolve()
                    if name not in RAW_FILES or not path.is_relative_to(directory) or not path.is_file():
                        self._json({"error": "Raw log not found"}, 404)
                        return
                    self._send(path.read_bytes(), "text/plain; charset=utf-8")
                    return
                self._json({"error": "Unknown route"}, 404)
            except (OSError, ValueError) as exc:
                self._json({"error": str(exc)}, 422)

        def do_POST(self) -> None:
            """Save human audit labels for a cataloged local run."""
            request = urlsplit(self.path)
            if request.path != "/api/annotations":
                self._json({"error": "Unknown route"}, 404)
                return
            host = self.headers.get("Host", "").split(":", 1)[0]
            origin = self.headers.get("Origin")
            if host not in {"127.0.0.1", "localhost"} or (origin and origin != f"http://{self.headers.get('Host')}"):
                self._json({"error": "Only same-origin local writes are allowed"}, 403)
                return
            run_id = parse_qs(request.query).get("run", [None])[0]
            directory = self._run_directory(run_id)
            if directory is None:
                self._json({"error": "Run not found in the active experiment catalog"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Annotation document must be between 1 byte and 1 MB")
                value = json.loads(self.rfile.read(size))
                self._json(save_annotations(directory, value))
            except (OSError, ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, 422)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_reviews(root: Path, port: int = 8765) -> None:
    """Run the local reviewer until interrupted."""
    with create_review_server(root, port) as server:
        print(f"Trajectory reviewer: http://127.0.0.1:{server.server_port}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
