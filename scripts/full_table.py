#!/usr/bin/env python3
"""
Ingest Aleph ILP distillation outputs into tidy CSVs + compute per-run Top-3 rules.

What it does (per run = dataset/teacher/preset):
- Reads two confusion matrices:
    * confusion_matrix_values_wrt_true_labels.txt   -> metrics vs truth
    * confusion_matrix_values_wrt_model_predictions.txt -> fidelity vs teacher
- Computes metrics: accuracy, balanced_accuracy, macro/weighted precision/recall/F1, MCC, fidelity
- Parses hypothesis rules (*.pl) to compute num_rules and avg_body_len
- Parses train_test_split_by_class.txt for n_train/n_test/n_classes
- Writes:
    * aleph_results.csv     (one row per run with metrics + complexity + Aleph settings)
    * aleph_confusion.csv   (long form, both bases = 'truth' and 'teacher')
    * aleph_rulebook.csv    (**Top-3 rules per run only**; columns:
                             dataset,teacher,preset,rule_id,head_class,body,body_len,support,coverage,precision)

Notes:
- All performance-like numbers we print (accuracy, f1, etc.) are formatted to 4 decimals as strings.
- In aleph_rulebook.csv: support is an integer (#pos covered + #neg covered);
  coverage is pos_covered / n_pos; precision is pos_covered / (pos_covered + neg_covered); both 4 d.p. strings.

Usage:
    python aleph_ingest.py --root /path/to/your/tree --outdir /path/to/outdir
If --outdir is omitted, outputs go into --root.

Expected tree:
  {root}/{dataset}/{teacher}/{preset}/
      confusion_matrix_values_wrt_true_labels.txt
      confusion_matrix_values_wrt_model_predictions.txt
      {dataset}_hypothesis.pl
      {dataset}_test.b
      {dataset}_test.f
      {dataset}_test.n
  {root}/{dataset}/{teacher}/train_test_split_by_class.txt
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple, Optional, Dict
import os
import csv

ALEPH_PRESETS = {
    "mushroom": {
        "dt": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 40).\n"
                    ":- aleph_set(nodes, 80000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(minacc, 0.90).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 1).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 64).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 8).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1000).\n"
                    ":- aleph_set(nodes, 150000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 20).\n"
                )
            },
        },
        "rf": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 40).\n"
                    ":- aleph_set(nodes, 80000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(minacc, 0.90).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 1).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 64).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 8).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1000).\n"
                    ":- aleph_set(nodes, 150000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 20).\n"
                )
            },
        },
        "xgb": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 40).\n"
                    ":- aleph_set(nodes, 80000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(minacc, 0.90).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 1).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 64).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 8).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1000).\n"
                    ":- aleph_set(nodes, 150000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 20).\n"
                )
            },
        },
    },
    "adult": {
        "dt": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 60).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(noise, 200).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 80).\n"
                    ":- aleph_set(nodes, 120000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 3).\n"
                    ":- aleph_set(noise, 400).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1500).\n"
                    ":- aleph_set(nodes, 200000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 2).\n"
                    ":- aleph_set(noise, 1400).\n"
                    ":- aleph_set(minacc, 0.60).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
        },
        "rf": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 60).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(noise, 200).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 80).\n"
                    ":- aleph_set(nodes, 120000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(noise, 400).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1500).\n"
                    ":- aleph_set(nodes, 200000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 3).\n"
                    ":- aleph_set(noise, 1400).\n"
                    ":- aleph_set(minacc, 0.60).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
        },
        "xgb": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 60).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(noise, 200).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 80).\n"
                    ":- aleph_set(nodes, 120000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(noise, 400).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1500).\n"
                    ":- aleph_set(nodes, 200000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 3).\n"
                    ":- aleph_set(noise, 1400).\n"
                    ":- aleph_set(minacc, 0.60).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
        },
    },
}

# ---------------------- Split-by-class parser ----------------------


def parse_split_by_class_txt(path: Path) -> Dict[str, Optional[int]]:
    if not path.exists():
        return {"n_train": None, "n_test": None, "n_classes": None}

    text = path.read_text().splitlines()
    mode = None  # "train" or "test"
    train_counts, test_counts = {}, {}
    for line in text:
        s = line.strip()
        if not s:
            continue
        sl = s.lower()
        if sl.startswith("training samples by class"):
            mode = "train"
            continue
        if sl.startswith("testing samples by class"):
            mode = "test"
            continue
        if s.lower() == "class":
            continue
        m = re.match(r"^([^\s]+)\s+(\d+)$", s)
        if m and mode in ("train", "test"):
            label, cnt = m.group(1), int(m.group(2))
            (train_counts if mode == "train" else test_counts)[label] = \
                (train_counts if mode == "train" else test_counts).get(label, 0) + cnt

    n_train = sum(train_counts.values()) if train_counts else None
    n_test = sum(test_counts.values()) if test_counts else None
    labels = set(train_counts) | set(test_counts)
    n_classes = len(labels) if labels else None
    return {"n_train": n_train, "n_test": n_test, "n_classes": n_classes}

# ---------------------- Metric formatting (4 d.p. strings) ----------------------


def fmt4(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def format_perf_metrics_inplace(d, keys=None):
    if keys is None:
        keys = [
            "accuracy", "balanced_accuracy",
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
            "mcc", "auroc", "fidelity"
        ]
    for k in keys:
        if k in d and d[k] is not None:
            d[k] = fmt4(d[k])

# ---------------------- Confusion matrix & metrics ----------------------


def cm_from_txt(path: Path) -> np.ndarray:
    text = path.read_text()
    lowered = text.lower()
    nums = [int(tok) for tok in re.findall(r'-?\d+', text)]
    if ('tp' in lowered and 'fp' in lowered and 'fn' in lowered and 'tn' in lowered) or len(nums) == 4:
        if len(nums) != 4:
            raise ValueError(
                f"Expected 4 numeric values for TP/FP/FN/TN in {path}, found {len(nums)}.")
        tp, fp, fn, tn = nums[0], nums[1], nums[2], nums[3]
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    rows = []
    for line in text.splitlines():
        row = [int(tok) for tok in re.findall(r'-?\d+', line)]
        if row:
            rows.append(row)
    if not rows:
        raise ValueError(f"No numbers found in {path}")

    ncols = min(len(r) for r in rows)
    arr = np.array([r[:ncols] for r in rows], dtype=int)
    n = min(arr.shape)
    cm = arr[:n, :n]
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"Confusion matrix is not square in {path}: {cm.shape}")
    return cm


def per_class_prf(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp),
                         where=(tp+fp) != 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn) != 0)
        f1 = np.divide(2*prec*rec, prec + rec,
                       out=np.zeros_like(tp), where=(prec+rec) != 0)
    return prec, rec, f1


def mcc_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    s = cm.sum()
    if s == 0:
        return float("nan")
    t_k = cm.sum(axis=0)
    p_k = cm.sum(axis=1)
    c = np.trace(cm)
    num = c * s - float((p_k * t_k).sum())
    term1 = s**2 - float((t_k**2).sum())
    term2 = s**2 - float((p_k**2).sum())
    if term1 < 0 and abs(term1) < 1e-12:
        term1 = 0.0
    if term2 < 0 and abs(term2) < 1e-12:
        term2 = 0.0
    denom = np.sqrt(term1 * term2)
    if denom == 0:
        return 0.0
    return float(num / denom)


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    n = cm.sum()
    k = cm.shape[0]
    acc = np.trace(cm) / n if n else np.nan
    prec, rec, f1 = per_class_prf(cm)
    macro_precision = float(np.nanmean(prec))
    macro_recall = float(np.nanmean(rec))
    macro_f1 = float(np.nanmean(f1))
    supports = cm.sum(axis=1)
    weights = supports / n if n else np.zeros_like(supports, dtype=float)
    weighted_precision = float(np.nansum(prec * weights))
    weighted_recall = float(np.nansum(rec * weights))
    weighted_f1 = float(np.nansum(f1 * weights))
    balanced_accuracy = macro_recall
    # MCC
    y_true = []
    for i in range(k):
        for j in range(k):
            c = int(cm[i, j])
            if c:
                y_true.extend([i]*c)
    mcc = mcc_from_cm(cm) if len(y_true) > 0 else np.nan
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc),
        "n_test": int(n),
        "n_classes": int(k),
    }

# ---------------------- Misc utils ----------------------


def read_json_if_exists(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

# --- DROP-IN: split on top-level commas (not inside parentheses) ---


def _split_top_level_commas(s: str) -> list:
    parts, buf, depth = [], "", 0
    for ch in (s or ""):
        if ch == '(':
            depth += 1
            buf += ch
        elif ch == ')':
            depth = max(0, depth - 1)
            buf += ch
        elif ch == ',' and depth == 0:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


# For complexity (count/length); keeps rule_id/body/body_len/head_class


# --- DROP-IN REPLACEMENT: parse_rules (correct body_len & body tokens) ---
def parse_rules(hypothesis_path: Path) -> List[dict]:
    """
    Parse hypothesis .pl into rules with head/body and body_len.
    Splits the body on top-level commas only (ignoring commas inside (...) ).
    Ignores comment lines starting with '%'.
    """
    if not hypothesis_path or not hypothesis_path.exists():
        return []
    txt = hypothesis_path.read_text()
    # strip % comments
    txt = "\n".join([ln for ln in txt.splitlines()
                    if not ln.strip().startswith('%')])

    # accumulate clauses until '.'
    clauses, buf = [], ""
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        buf += (" " if buf else "") + ln
        if ln.endswith('.'):
            clauses.append(buf)
            buf = ""

    rules, rid = [], 0
    for cl in clauses:
        cl = cl.rstrip('.').strip()
        if not cl:
            continue
        if ":-" in cl:
            head, body = cl.split(":-", 1)
            body_parts = _split_top_level_commas(body)
            body_norm = ", ".join(body_parts)
            body_len = len(body_parts)
        else:
            head, body_norm, body_len = cl, "", 0

        head = head.strip()
        m = re.match(r'([a-zA-Z_]\w*)\s*\(([^()]*)\)', head)
        head_class = m.group(1) if m else re.sub(r'\(.*', '', head)

        rid += 1
        rules.append({
            "rule_id": rid,
            "head_class": head_class,
            "body": body_norm,
            "body_len": body_len
        })
    return rules


def parse_aleph_settings(settings_str: str) -> dict:
    kv = {}
    for line in settings_str.strip().splitlines():
        m = re.search(r'aleph_set\(([^,]+),\s*([^)]+)\)', line)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip().strip('.').strip("'").strip('"')
        if re.fullmatch(r'-?\d+', val):
            val_cast = int(val)
        elif re.fullmatch(r'-?\d+\.\d*', val):
            val_cast = float(val)
        else:
            val_cast = val
        kv[key] = val_cast
    return kv


def parse_aleph_cfg(dataset: str, teacher: str, preset: str) -> dict:
    try:
        settings = ALEPH_PRESETS[dataset][teacher][preset]["settings"]
        kv = parse_aleph_settings(settings)
    except Exception:
        kv = {}
    keep = ["search", "openlist", "nodes", "evalfn",
            "clauselength", "minacc", "minpos", "noise"]
    for k in keep:
        kv.setdefault(k, None)
    return kv


def collect_runs(root: Path) -> List[Tuple[str, str, str, Path]]:
    runs = []
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for teacher_dir in dataset_dir.iterdir():
            if not teacher_dir.is_dir():
                continue
            teacher = teacher_dir.name
            for preset_dir in teacher_dir.iterdir():
                if not preset_dir.is_dir():
                    continue
                preset = preset_dir.name
                cm_truth = preset_dir / "confusion_matrix_values_wrt_true_labels.txt"
                cm_teacher = preset_dir / "confusion_matrix_values_wrt_model_predictions.txt"
                eval_json = preset_dir / "evaluation_results.json"
                hyp_pl = next(preset_dir.glob("*_hypothesis.pl"), None)
                if cm_truth.exists() or cm_teacher.exists() or eval_json.exists() or hyp_pl:
                    runs.append((dataset, teacher, preset, preset_dir))
    return runs

# ---------------------- Rule firing utilities (for Top-3) ----------------------


# --- DROP-IN REPLACEMENT: _parse_rules_atoms used by Top-3 computation ---
def _parse_rules_atoms(rule_file: str) -> List[Tuple[str, List[str]]]:
    """
    Return list of (head_functor, [cond,...]) from hypothesis file.
    Splits on top-level commas only.
    """
    p = Path(rule_file)
    if not p.exists():
        return []
    txt = p.read_text()
    txt = "\n".join([ln for ln in txt.splitlines()
                    if not ln.strip().startswith('%')])

    # collect full clauses ending with '.'
    clauses, buf = [], ""
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        buf += (" " if buf else "") + ln
        if ln.endswith('.'):
            clauses.append(buf)
            buf = ""

    rules = []
    for cl in clauses:
        cl = cl.rstrip('.').strip()
        if not cl:
            continue
        if ":-" in cl:
            head, body = cl.split(":-", 1)
            body_parts = _split_top_level_commas(body)
        else:
            head, body_parts = cl, []

        m = re.match(r'([a-zA-Z_]\w*)\s*\(([^()]*)\)', head.strip())
        head_functor = m.group(1) if m else re.sub(r'\(.*', '', head.strip())
        rules.append((head_functor, body_parts))
    return rules


# --- OPTIONAL DROP-IN (minor hardening): strip quotes on example ids in features/examples ---

def load_features(b_path: str) -> Dict[str, set]:
    feats: Dict[str, set] = {}
    p = Path(b_path)
    if not p.exists():
        return feats
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith('%') or not s.endswith('.'):
            continue
        m = re.match(r'^([a-zA-Z_]\w*)\s*\((.*)\)\.$', s)
        if not m:
            continue
        pred, args_str = m.group(1), m.group(2)
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        if not args:
            continue
        ex_id = args[0].strip("'\"")
        rest = args[1:]
        key = f"{pred}({','.join(rest)})" if rest else f"{pred}()"
        feats.setdefault(ex_id, set()).add(key)
    return feats


def load_examples(ex_path: str) -> List[Tuple[str, str]]:
    exs: List[Tuple[str, str]] = []
    p = Path(ex_path)
    if not p.exists():
        return exs
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith('%') or not s.endswith('.'):
            continue
        m = re.match(r'^([a-zA-Z_]\w*)\s*\(\s*([^(),\s]+)\s*\)\.$', s)
        if not m:
            continue
        pred, ex_id = m.group(1), m.group(2).strip("'\"")
        exs.append((pred, ex_id))
    return exs


# --- DROP-IN: make _cond_to_key a bit more robust ---
def _cond_to_key(cond: str) -> Optional[str]:
    """
    'v_education(A, v_below_hs)' -> 'v_education(v_below_hs)'  (drop first arg)
    """
    s = cond.strip().rstrip('.')
    m = re.match(r'^([a-zA-Z_]\w*)\s*\((.*)\)$', s)
    if not m:
        return None
    pred = m.group(1)
    # split the argument list (no nested parens expected inside args)
    args = [a.strip() for a in m.group(2).split(',') if a.strip()]
    # drop the first (example id); keep the rest
    rest = args[1:] if len(args) >= 1 else []
    return f"{pred}({','.join(rest)})"


def predicts(rules: List[Tuple[str, List[str]]], example_id: str, features: Dict[str, set], target_pred: str) -> bool:
    """
    Returns True if ANY of the given rules fires for this example (conjunctive bodies).
    """
    feat_set = features.get(example_id, set())
    for head_functor, conds in rules:
        if head_functor != target_pred:
            continue
        ok = True
        for c in conds:
            key = _cond_to_key(c)
            if key is None or key not in feat_set:
                ok = False
                break
        if ok:
            return True
    return False


# --- DROP-IN REPLACEMENT: _index_rule_defs_with_keys (use safe split) ---
def _index_rule_defs_with_keys(hypothesis_path: Path) -> List[dict]:
    """
    Build an index of rules with canonical 'keys' (sorted cond keys) for id/body lookup.
    """
    defs = parse_rules(hypothesis_path)
    out = []
    for r in defs:
        parts = _split_top_level_commas(r["body"]) if r["body"] else []
        keys = tuple(sorted([_cond_to_key(p) or p for p in parts]))
        out.append({**r, "keys": keys})
    return out


def _find_rule_id_and_body(head_functor: str, conds: List[str], idx: List[dict]) -> Tuple[Optional[int], str, Optional[int]]:
    keys = sorted([_cond_to_key(c) or c for c in conds])
    for r in idx:
        if r["head_class"] == head_functor and r["keys"] == tuple(keys):
            return r["rule_id"], r["body"], r["body_len"]
    # fallback if not found
    return None, ", ".join(conds), len(conds)


def compute_top3_rows_for_run(dataset: str, teacher: str, preset: str, run_dir: Path) -> List[dict]:
    """
    Compute Top-3 rules per run and return rows with requested columns.
    Columns: dataset,teacher,preset,rule_id,head_class,body,body_len,support,coverage,precision
    """
    rule_file = run_dir / f"{dataset}_hypothesis.pl"
    b_file = run_dir / f"{dataset}_test.b"
    fpos_file = run_dir / f"{dataset}_test.f"
    fneg_file = run_dir / f"{dataset}_test.n"

    rules_atoms = _parse_rules_atoms(str(rule_file))
    features = load_features(str(b_file))
    pos_examples = load_examples(str(fpos_file))
    neg_examples = load_examples(str(fneg_file))

    if not pos_examples:
        return []  # nothing to compute

    target_pred = pos_examples[0][0]
    n_pos = len(pos_examples)

    # compute pos/neg coverage per rule
    stats = []
    for head_functor, conds in rules_atoms:
        if head_functor != target_pred:
            continue
        pos_cov = sum(1 for _, ex in pos_examples if predicts(
            [(head_functor, conds)], ex, features, target_pred))
        neg_cov = sum(1 for _, ex in neg_examples if predicts(
            [(head_functor, conds)], ex, features, target_pred))
        support = pos_cov + neg_cov
        coverage = (pos_cov / n_pos) if n_pos else 0.0
        precision = (pos_cov / support) if support > 0 else 0.0
        stats.append(((head_functor, conds), pos_cov,
                     neg_cov, support, coverage, precision))

    # pick Top-3 by positive coverage desc, then precision desc, then shorter body
    stats.sort(key=lambda x: (x[1], x[5], -len(x[0][1])), reverse=True)
    idx = _index_rule_defs_with_keys(rule_file)

    rows = []
    for (head_functor, conds), pos_cov, neg_cov, support, coverage, precision in stats[:3]:
        rule_id, body, body_len = _find_rule_id_and_body(
            head_functor, conds, idx)
        rows.append({
            "dataset": dataset,
            "teacher": teacher,
            "preset": preset,
            "rule_id": "" if rule_id is None else int(rule_id),
            "head_class": head_functor,
            "body": body,
            "body_len": int(body_len) if body_len is not None else len(conds),
            "support": int(support),
            "coverage": fmt4(coverage),   # 4 decimals (as string)
            "precision": fmt4(precision),  # 4 decimals (as string)
        })
    return rows

# ---------------------- MAIN ----------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory of your results tree")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Where to write CSVs (defaults to --root)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows, confusion_rows, rulebook_rows = [], [], []

    for dataset, teacher, preset, run_dir in collect_runs(root):
        # ----- split counts -----
        split_path = run_dir.parent / "train_test_split_by_class.txt"
        split_stats = parse_split_by_class_txt(split_path)

        # ----- confusions -----
        cm_truth_path = run_dir / "confusion_matrix_values_wrt_true_labels.txt"
        cm_teacher_path = run_dir / "confusion_matrix_values_wrt_model_predictions.txt"

        truth_metrics, fidelity = {}, None
        if cm_truth_path.exists():
            cm_t = cm_from_txt(cm_truth_path)
            truth_metrics = metrics_from_cm(cm_t)
            format_perf_metrics_inplace(truth_metrics)
            for i in range(cm_t.shape[0]):
                for j in range(cm_t.shape[1]):
                    confusion_rows.append({
                        "dataset": dataset, "teacher": teacher, "preset": preset,
                        "basis": "truth", "actual": i, "predicted": j, "count": int(cm_t[i, j])
                    })
        if cm_teacher_path.exists():
            cm_teach = cm_from_txt(cm_teacher_path)
            teach_metrics = metrics_from_cm(cm_teach)
            format_perf_metrics_inplace(teach_metrics)
            fidelity = teach_metrics.get("accuracy")
            for i in range(cm_teach.shape[0]):
                for j in range(cm_teach.shape[1]):
                    confusion_rows.append({
                        "dataset": dataset, "teacher": teacher, "preset": preset,
                        "basis": "teacher", "actual": i, "predicted": j, "count": int(cm_teach[i, j])
                    })

        # ----- complexity (num_rules/avg_body_len) -----
        hyp_pl = next(run_dir.glob("*_hypothesis.pl"), None)
        rules = parse_rules(hyp_pl) if hyp_pl else []
        num_rules = len(rules) if rules else None
        avg_body_len = fmt4(float(np.mean(
            [r["body_len"] for r in rules])) if rules else np.nan) if rules else None

        # ----- aleph settings -----
        aleph_cfg = parse_aleph_cfg(dataset, teacher, preset)

        # ----- summary row -----
        row = {
            "dataset": dataset, "teacher": teacher, "preset": preset,
            **{k: aleph_cfg.get(k) for k in ["search", "openlist", "nodes", "evalfn", "clauselength", "minacc", "minpos", "noise"]},
            "accuracy":           truth_metrics.get("accuracy"),
            "balanced_accuracy":  truth_metrics.get("balanced_accuracy"),
            "macro_precision":    truth_metrics.get("macro_precision"),
            "macro_recall":       truth_metrics.get("macro_recall"),
            "macro_f1":           truth_metrics.get("macro_f1"),
            "weighted_precision": truth_metrics.get("weighted_precision"),
            "weighted_recall":    truth_metrics.get("weighted_recall"),
            "weighted_f1":        truth_metrics.get("weighted_f1"),
            "mcc":                truth_metrics.get("mcc"),
            "fidelity":           fidelity,
            "num_rules":          num_rules,
            "avg_body_len":       avg_body_len,
            "n_train":            split_stats.get("n_train"),
            "n_test":             split_stats.get("n_test") if split_stats.get("n_test") is not None
            else (metrics_from_cm(cm_from_txt(cm_truth_path))["n_test"] if cm_truth_path.exists() else None),
            "n_classes":          split_stats.get("n_classes"),
        }
        summary_rows.append(row)

        # ----- TOP-3 RULES per run -> rulebook_rows (ONLY these) -----
        rulebook_rows.extend(
            compute_top3_rows_for_run(dataset, teacher, preset, run_dir)
        )

    # ----- write outputs -----
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "teacher", "preset"]).reset_index(drop=True)
    confusion_df = pd.DataFrame(confusion_rows)

    # IMPORTANT: rulebook only contains the Top-3 rows per run with the exact requested columns/order
    rulebook_df = pd.DataFrame(rulebook_rows, columns=[
        "dataset", "teacher", "preset", "rule_id", "head_class", "body", "body_len", "support", "coverage", "precision"
    ])

    out_summary = outdir / "aleph_results.csv"
    out_confusion = outdir / "aleph_confusion.csv"
    out_rulebook = outdir / "aleph_rulebook.csv"

    summary_df.to_csv(out_summary, index=False)
    confusion_df.to_csv(out_confusion, index=False)
    rulebook_df.to_csv(out_rulebook, index=False)

    print(f"Wrote:\n  {out_summary}\n  {out_confusion}\n  {out_rulebook}")

# ---------------------- CLI ----------------------


if __name__ == "__main__":
    main()
