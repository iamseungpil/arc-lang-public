# GPT-OSS 20B ARC Evaluation - Detailed Results

## Overall Summary

| Category | Count | Percentage |
|----------|------:|----------:|
| Total tasks in evaluation set | 400 | 100.00% |
| Tasks attempted | 146 | 36.50% |
| Tasks not attempted | 254 | 63.50% |

## Attempted Tasks Breakdown

| Category | Count | Percentage of Attempted | Percentage of Total |
|----------|------:|----------------------:|-------------------:|
| ✅ Correct solutions | 96 | 65.75% | 24.00% |
| ❌ Wrong output (valid format) | 50 | 34.25% | 12.50% |
| ⚠️ Invalid format/Empty output | 0 | 0.00% | 0.00% |

## Accuracy Metrics

- **Accuracy on attempted tasks**: 96/146 = **65.75%**
- **Accuracy on full evaluation set**: 96/400 = **24.00%**

## Task Status Distribution

| Status | Description | Count |
|--------|-------------|------:|
| ✅ Solved | Correct solution generated | 96 |
| ❌ Wrong | Valid grid but incorrect solution | 50 |
| ⏳ Not Attempted | Task not yet processed | 254 |
| ⚠️ Failed | Invalid format or empty output | 0 |

## Notes

- The experiment is still running (145→146 tasks completed so far)
- No tasks produced invalid format or empty outputs (100% valid grid generation)
- All 146 attempted tasks produced valid grid outputs
- Main failure mode: correct format but wrong transformation logic (50 tasks)
