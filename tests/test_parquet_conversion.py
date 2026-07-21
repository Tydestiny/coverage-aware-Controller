import json

import pyarrow as pa
import pyarrow.parquet as pq

from agent_error_compaction.parquet_utils import convert_parquet_files


def test_converts_multiple_parquet_files_to_jsonl(tmp_path):
    first = tmp_path / "safe.parquet"
    second = tmp_path / "unsafe.parquet"
    pq.write_table(pa.Table.from_pylist([{"conv_id": "safe", "turns": [{"content": "ok"}]}]), first)
    pq.write_table(pa.Table.from_pylist([{"conv_id": "unsafe", "turns": [{"content": "bad"}], "mistake_step": 0}]), second)

    output = tmp_path / "records.jsonl"
    manifest = convert_parquet_files([first, second], output, batch_size=1)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["conv_id"] for record in records] == ["safe", "unsafe"]
    assert records[1]["mistake_step"] == 0
    assert manifest["total_rows"] == 2
    assert manifest["sources"][0]["row_count"] == 1


def test_rejects_non_parquet_input(tmp_path):
    source = tmp_path / "data.json"
    source.write_text("{}", encoding="utf-8")

    try:
        convert_parquet_files([source], tmp_path / "output.jsonl")
    except ValueError as exc:
        assert ".parquet" in str(exc)
    else:
        raise AssertionError("non-Parquet input should be rejected")

