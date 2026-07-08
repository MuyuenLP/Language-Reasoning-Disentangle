# MLRC 2026 Reproduction Log

## Paper

**Title:** When Less Language is More: Investigating Language-Reasoning Disentanglement in Multilingual Reasoning Models

**Goal:** Reproduce the results of the paper and document all reproducibility issues.

---

# Environment

## Hardware

- GPU: NVIDIA L4 (24 GB)

## Software

- Python: 3.12.11
- PyTorch: 2.5.1
- Transformers: 4.51.3
- vLLM: 0.7.3
- CUDA: 12.4

---

# Progress

## ✅ Environment

- Repository cloned
- MLRS installed
- FreeEvalLM installed
- HuggingFace login successful

---

## ✅ Model

- DeepSeek-R1-Distill-Qwen-7B downloaded
- Successfully loaded with Transformers
- Successfully loaded with vLLM

---

## ✅ First Successful Steering Run

Task: MGSM

Sample Size: 2

Steering Strength:

[0.3, -0.1]

Output:

results/debug_mgsm/

Status:

SUCCESS

---

# Reproducibility Issues Found

## Issue 1

Missing dependency:

math_verify

Status:

Fixed

---

## Issue 2

Missing language identification model:

glotlid/model.bin

Status:

Downloaded manually

---

## Issue 3

Hard-coded absolute path:

/home/jhguo/...

Status:

Patched

---

## Issue 4

Dataset loader assumes incorrect working directory.

Status:

Patched

---

# Current Status

Environment ✅

Pipeline ✅

First Experiment ✅

Baseline ❌ (Next)