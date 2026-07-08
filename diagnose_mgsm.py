#!/usr/bin/env python3
"""
MGSM Reproduction Diagnostic Script
Checks for: per-language accuracy, garbled/corrupted text detection,
and mixed-language leakage in model responses.

Usage:
    python3 diagnose_mgsm.py ./results/baseline_mgsm
    python3 diagnose_mgsm.py ./results/debug_mgsm
"""

import json
import re
import sys
import os
import glob
from collections import defaultdict

# Characters that indicate decoding corruption (box-drawing, block elements, replacement char)
CORRUPT_CHAR_RANGES = [
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0xFFFD, 0xFFFD),  # Replacement character
]

def is_corrupt_char(ch):
    cp = ord(ch)
    for lo, hi in CORRUPT_CHAR_RANGES:
        if lo <= cp <= hi:
            return True
    return False

def corruption_ratio(text):
    if not text:
        return 0.0
    corrupt = sum(1 for ch in text if is_corrupt_char(ch))
    return corrupt / max(len(text), 1)

def detect_script_mixing(text, expected_lang):
    han = len(re.findall(r'[\u4e00-\u9fff]', text))
    thai = len(re.findall(r'[\u0e00-\u0e7f]', text))
    telugu = len(re.findall(r'[\u0c00-\u0c7f]', text))
    flags = []
    if expected_lang not in ("zh",) and han > 5:
        flags.append(f"unexpected_han({han})")
    if expected_lang not in ("th",) and thai > 5:
        flags.append(f"unexpected_thai({thai})")
    if expected_lang not in ("te",) and telugu > 5:
        flags.append(f"unexpected_telugu({telugu})")
    return flags

def analyze_file(path):
    lang = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"lang": lang, "error": str(e)}

    n = len(data)
    scores = [d.get("score", None) for d in data]
    valid_scores = [s for s in scores if s is not None]
    acc = sum(valid_scores) / len(valid_scores) if valid_scores else None

    corrupt_examples = []
    mixing_examples = []
    invalid_extractions = 0

    for i, d in enumerate(data):
        resp = d.get("response", "") or ""
        cr = corruption_ratio(resp)
        if cr > 0.02:
            corrupt_examples.append((i, round(cr, 4)))

        mix_flags = detect_script_mixing(resp, lang)
        if mix_flags:
            mixing_examples.append((i, mix_flags))

        fa = d.get("filtered_answer", None)
        if fa is None or fa == ["[invalid]"] or fa == "[invalid]":
            invalid_extractions += 1

    return {
        "lang": lang,
        "n": n,
        "accuracy": round(acc, 4) if acc is not None else None,
        "num_corrupt_responses": len(corrupt_examples),
        "corrupt_examples_idx": corrupt_examples[:5],
        "num_script_mixing": len(mixing_examples),
        "mixing_examples_idx": mixing_examples[:5],
        "num_invalid_extractions": invalid_extractions,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_mgsm.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    files = [f for f in files if os.path.basename(f) not in ("Results.csv",)]

    if not files:
        print(f"No .json files found in {results_dir}")
        sys.exit(1)

    print(f"\n{'='*90}")
    print(f"MGSM DIAGNOSTIC REPORT — {results_dir}")
    print(f"{'='*90}\n")

    all_results = [analyze_file(f) for f in files]

    header = f"{'Lang':<6} {'N':<5} {'Acc':<8} {'#Corrupt':<10} {'#ScriptMix':<12} {'#InvalidExtr':<14}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if "error" in r:
            print(f"{r['lang']:<6} ERROR: {r['error']}")
            continue
        print(f"{r['lang']:<6} {r['n']:<5} {str(r['accuracy']):<8} {r['num_corrupt_responses']:<10} "
              f"{r['num_script_mixing']:<12} {r['num_invalid_extractions']:<14}")

    print(f"\n{'='*90}")
    print("FLAGGED EXAMPLES (corruption or script mixing detected)")
    print(f"{'='*90}\n")

    for r in all_results:
        if "error" in r:
            continue
        if r["num_corrupt_responses"] > 0 or r["num_script_mixing"] > 0:
            print(f"--- {r['lang']} ---")
            if r["corrupt_examples_idx"]:
                print(f"  Corrupt response indices (idx, corrupt_ratio): {r['corrupt_examples_idx']}")
            if r["mixing_examples_idx"]:
                print(f"  Script-mixing indices (idx, flags): {r['mixing_examples_idx']}")
            print()

    valid_accs = [r["accuracy"] for r in all_results if r.get("accuracy") is not None]
    overall_acc = sum(valid_accs) / len(valid_accs) if valid_accs else None
    total_corrupt = sum(r.get("num_corrupt_responses", 0) for r in all_results if "error" not in r)
    total_mixing = sum(r.get("num_script_mixing", 0) for r in all_results if "error" not in r)
    total_invalid = sum(r.get("num_invalid_extractions", 0) for r in all_results if "error" not in r)
    total_n = sum(r.get("n", 0) for r in all_results if "error" not in r)

    print(f"{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"Overall accuracy (macro-avg across languages): {round(overall_acc, 4) if overall_acc else 'N/A'}")
    print(f"Total examples: {total_n}")
    print(f"Total corrupted responses: {total_corrupt} ({round(100*total_corrupt/max(total_n,1),2)}%)")
    print(f"Total script-mixing responses: {total_mixing} ({round(100*total_mixing/max(total_n,1),2)}%)")
    print(f"Total invalid/unextractable answers: {total_invalid} ({round(100*total_invalid/max(total_n,1),2)}%)")
    print()

    broken_langs = [r["lang"] for r in all_results if "error" not in r and r.get("accuracy") == 0.0]
    high_corruption_langs = [r["lang"] for r in all_results if "error" not in r and r["n"] > 0 and r["num_corrupt_responses"] / r["n"] > 0.3]

    if broken_langs:
        print(f"WARNING: Languages with 0% accuracy: {broken_langs}")
    if high_corruption_langs:
        print(f"WARNING: Languages with >30% corrupted responses: {high_corruption_langs}")
    if not broken_langs and not high_corruption_langs:
        print("No major corruption or zero-accuracy languages detected.")

    print()

if __name__ == "__main__":
    main()