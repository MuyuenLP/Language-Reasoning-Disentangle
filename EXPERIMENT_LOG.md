# Experiment Log

---

## Experiment 001

### Date

2026-07-03

### Model

DeepSeek-R1-Distill-Qwen-7B

### Task

MGSM

### Sample

2

### Steering

[0.3, -0.1]

### Results

| Language | Score | Response Fidelity | Reasoning Fidelity |
|-----------|------:|------------------:|-------------------:|
| en | 1.0 | 0.5 | 1.0 |
| te | 0.0 | 1.0 | 1.0 |
| fr | 1.0 | 0.0 | 1.0 |
| sw | 0.0 | 0.5 | 0.5 |
| bn | 0.5 | 0.0 | 1.0 |
| ja | 1.0 | 0.5 | 1.0 |
| th | 0.0 | 1.0 | 0.5 |
| ru | 0.5 | 0.5 | 1.0 |
| zh | 1.0 | 1.0 | 1.0 |
| es | 1.0 | 0.0 | 1.0 |
| de | 0.5 | 0.0 | 1.0 |

### Final

Accuracy: 0.5909

Response Fidelity: 0.4545

Reasoning Fidelity: 0.9091

### Notes

First successful end-to-end steering experiment.

Experiment 002
--------------
Task: MGSM
Sample: 2
Steering: [0,0]

Final Accuracy: 0.6364
Response Fidelity: 0.4545
Reasoning Fidelity: 0.9091

Purpose:
Baseline for comparison with steering.

## Experiment 002

Task: MGSM (Full)

Model: DeepSeek-R1-Distill-Qwen-7B

Method: Baseline (No Steering)

Steering Strength:
[0,0]

Dataset:
2750 examples

Results

Accuracy: 0.3684

Response Fidelity: 0.5389

Reasoning Fidelity: 0.7513

Status:
SUCCESS

## Experiment 003

Task: MGSM (Full)

Model:
DeepSeek-R1-Distill-Qwen-7B

Method:
MLRS Steering

Steering Strength:
[0.3, -0.1]

Results

Accuracy: 0.3651

Response Fidelity: 0.5865

Reasoning Fidelity: 0.8989

Comparison with baseline

Accuracy Δ = -0.0033
Response Fidelity Δ = +0.0476
Reasoning Fidelity Δ = +0.1476

Status:
SUCCESS