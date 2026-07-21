# Dataset Report

- Trajectories: 2276
- Prefixes: 12964
- Family leakage check: passed
- Rejected records: 0

## Distributions

```json
{
  "distance_to_error_onset_distribution": {
    "-1": 1114,
    "-2": 950,
    "-3": 803,
    "0": 1114,
    "1": 1015,
    "2": 810,
    "3": 676,
    "<missing>": 6482
  },
  "domain_distribution": {
    "agentic": 875,
    "coding": 608,
    "math": 793
  },
  "error_occurred_distribution": {
    "False": 9349,
    "True": 3615
  },
  "outcome_distribution": {
    "failure": 1114,
    "success": 1162
  },
  "unsafe_source_distribution": {
    "diagnosed": 386,
    "injected": 728,
    "safe": 1162
  }
}
```

## Split Summary

```json
{
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
}
```
