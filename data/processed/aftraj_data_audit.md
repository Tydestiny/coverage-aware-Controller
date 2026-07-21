# AFTraj-2K Data Audit

## Parquet conversion

- Source files: 2
- Source rows: 2276
- JSONL output: `data\raw\aftraj.jsonl`
- Streaming batch size: 256

| Source | Rows | SHA-256 |
|---|---:|---|
| `dataset\AFTraj-2K\aftraj_safe.parquet` | 1162 | `cd16e0de24fa9f70f9832c7fee86ed2370d4d42deb8bdb6b97939c323aa8daa3` |
| `dataset\AFTraj-2K\aftraj_unsafe.parquet` | 1114 | `f19d407898ebf32b32a2e99600f49f53a358ac2831fe437da3dc2a9fe25f8f41` |

## Source format inspection

- Trajectories: 2276
- Safe: 1162
- Failure: 1114
- `mistake_step` range: validated during conversion to validated during conversion
- Selected convention: 0-based
- Out of bounds under selected convention: 0 rejected by canonical validation

Observed source schema:

```json
{
  "field_types": {},
  "parquet_schema": [
    "conv_id: string\ndomain: string\ntask: string\ngold_answer: string\nnum_turns: int64\ntools: list<element: struct<description: string, name: string>>\n  child 0, element: struct<description: string, name: string>\n      child 0, description: string\n      child 1, name: string\nturns: list<element: struct<action: string, content: string, role: string, thought: string>>\n  child 0, element: struct<action: string, content: string, role: string, thought: string>\n      child 0, action: string\n      child 1, content: string\n      child 2, role: string\n      child 3, thought: string\n-- schema metadata --\npandas: '{\"index_columns\": [], \"column_indexes\": [], \"columns\": [{\"name\":' + 894",
    "conv_id: string\ndomain: string\ntask: string\ngold_answer: string\nnum_turns: int64\ntools: list<element: struct<description: string, name: string>>\n  child 0, element: struct<description: string, name: string>\n      child 0, description: string\n      child 1, name: string\nturns: list<element: struct<action: string, content: string, role: string, thought: string>>\n  child 0, element: struct<action: string, content: string, role: string, thought: string>\n      child 0, action: string\n      child 1, content: string\n      child 2, role: string\n      child 3, thought: string\nmistake_step: int64\nmistake_agent: string\nmistake_reason: string\nunsafe_source: string\n-- schema metadata --\npandas: '{\"index_columns\": [], \"column_indexes\": [], \"columns\": [{\"name\":' + 1395"
  ]
}
```

## Canonical validation

- Canonical trajectories: 2276
- Rejected records: 0
- Family leakage: PASS
- Distinct families: 1728
- Prefix samples: 12964

Rejected reason counts:

```json
{}
```

## Distribution summary

```json
{
  "distance_to_error_onset": {
    "-1": 1114,
    "-2": 950,
    "-3": 803,
    "0": 1114,
    "1": 1015,
    "2": 810,
    "3": 676,
    "<missing>": 6482
  },
  "domain": {
    "agentic": 875,
    "coding": 608,
    "math": 793
  },
  "error_occurred": {
    "False": 9349,
    "True": 3615
  },
  "error_onset_absolute": {
    "1": 164,
    "10": 58,
    "11": 10,
    "12": 15,
    "13": 13,
    "14": 7,
    "15": 6,
    "16": 7,
    "17": 1,
    "18": 5,
    "19": 4,
    "2": 147,
    "20": 2,
    "21": 1,
    "22": 1,
    "24": 3,
    "28": 1,
    "3": 311,
    "30": 1,
    "31": 1,
    "33": 1,
    "4": 141,
    "5": 22,
    "6": 35,
    "65": 1,
    "7": 54,
    "8": 30,
    "9": 72
  },
  "outcome": {
    "failure": 1114,
    "success": 1162
  },
  "responsible_actor": {
    "CodeTester": 118,
    "CodeWriter": 129,
    "Geography_Expert": 18,
    "Manager": 238,
    "MathSolver": 310,
    "Movie_Expert": 9,
    "Research_Expert": 17,
    "Statistics_Expert": 1,
    "TaskSolver": 42,
    "Verification_Expert": 3,
    "Verifier": 84,
    "environment": 4,
    "search_agent": 141
  },
  "splits": {
    "test": {
      "domain_distribution": {
        "agentic": 176,
        "coding": 122,
        "math": 158
      },
      "outcome_distribution": {
        "failure": 224,
        "success": 232
      },
      "trajectory_count": 456
    },
    "train": {
      "domain_distribution": {
        "agentic": 525,
        "coding": 365,
        "math": 476
      },
      "outcome_distribution": {
        "failure": 668,
        "success": 698
      },
      "trajectory_count": 1366
    },
    "validation": {
      "domain_distribution": {
        "agentic": 174,
        "coding": 121,
        "math": 159
      },
      "outcome_distribution": {
        "failure": 222,
        "success": 232
      },
      "trajectory_count": 454
    }
  },
  "trajectory_length": {
    "max": 74,
    "mean": 11.557996485061512,
    "median": 11.0,
    "min": 3,
    "p25": 7,
    "p75": 13,
    "p90": 20
  },
  "unsafe_source": {
    "diagnosed": 386,
    "injected": 728,
    "safe": 1162
  }
}
```

## Audit conclusions

- Parquet rows are converted one-for-one to JSONL records in source-file order.
- Canonical conversion preserves one step per source turn and retains each complete `raw_step`.
- Family grouping is completed before split assignment; the report performs a strict leakage check.
- Prefixes contain only steps up to and including `prefix_end_step`.
- Error reasons and source labels are preserved without inference or rewriting.
