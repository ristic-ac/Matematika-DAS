import os
import re
from typing import List, Dict, Tuple, Set, Iterable, Mapping, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
import json
from statistics import mean

# A feature condition is now (predicate, value)
Feature = Tuple[str, str]
Rule = Tuple[str, List[Feature]]

def parse_rules(rule_file: str) -> List[Rule]:
    """
    Parse rules of the form:
      target(A) :- color(A, red), shape(A, circle).
    """
    rules: List[Rule] = []
    head_pat = re.compile(r"^(\w+)\s*\(\s*\w+\s*\)\s*:-\s*(.*)\.$")
    # body literals are strictly binary: pred(Var, Val)
    cond_pat = re.compile(r"(\w+)\s*\(\s*\w+\s*,\s*([^\)]+)\s*\)")
    with open(rule_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            m_head = head_pat.match(line)
            if not m_head:
                continue
            target = m_head.group(1)
            body = m_head.group(2)
            conds = [(p.strip(), v.strip()) for (p, v) in cond_pat.findall(body)]
            rules.append((target, conds))
    return rules

def load_features(b_file: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Load background facts of the form:
      color(x1, red).
      shape(x1, circle).
    Returns: { ex_id: { pred: {val1, val2, ...} } }
    """
    feats: Dict[str, Dict[str, Set[str]]] = {}
    pat = re.compile(r"(\w+)\s*\(\s*(\w+)\s*,\s*([^\)]+)\s*\)\s*\.")
    with open(b_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            m = pat.match(line)
            if not m:
                continue
            pred, ex, val = m.groups()
            pred, ex, val = pred.strip(), ex.strip(), val.strip()
            feats.setdefault(ex, {}).setdefault(pred, set()).add(val)
    return feats

def load_examples(f_file: str) -> List[Tuple[str, str]]:
    """
    Examples remain unary:
      target(x1).
      target(x2).
    """
    exs: List[Tuple[str, str]] = []
    pat = re.compile(r"(\w+)\s*\(\s*(\w+)\s*\)\s*\.")
    with open(f_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            m = pat.match(line)
            if m:
                exs.append((m.group(1).strip(), m.group(2).strip()))
    return exs

def predicts(rules: List[Rule], ex_id: str,
             features: Dict[str, Dict[str, Set[str]]],
             target_pred: str) -> bool:
    """
    A rule fires iff ALL its conds (pred, val) are present in features for ex_id.
    """
    if ex_id not in features:
        return False
    fdict = features[ex_id]  # {pred: {values}}
    for (head, conds) in rules:
        if head != target_pred:
            continue
        if all(pred in fdict and val in fdict[pred] for pred, val in conds):
            return True
    return False

def compute_fidelity_confusion_for_folder(folder: str, dataset: str) -> Dict[str, float]:
    rule_file = os.path.join(folder, f"{dataset}_hypothesis.pl")
    b_file = os.path.join(folder, f"{dataset}_test.b")
    fpos_file = os.path.join(folder, f"{dataset}_test.f")
    fneg_file = os.path.join(folder, f"{dataset}_test.n")

    rules = parse_rules(rule_file)
    features = load_features(b_file)
    pos_examples = load_examples(fpos_file)
    neg_examples = load_examples(fneg_file)

    if not pos_examples:
        raise ValueError("No positive examples found.")
    target_pred = pos_examples[0][0]

    TP = FP = FN = TN = 0
    for _, ex in pos_examples:
        pred_val = predicts(rules, ex, features, target_pred)
        if pred_val:
            TP += 1
        else:
            FN += 1

    for _, ex in neg_examples:
        pred_val = predicts(rules, ex, features, target_pred)
        if pred_val:
            FP += 1
        else:
            TN += 1

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else 0.0

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "Accuracy": acc}

def print_confusion_matrix_distillate(metrics: Dict[str, int], outdir: str, suffix):
    TP = metrics["TP"]
    FP = metrics["FP"]
    FN = metrics["FN"]
    TN = metrics["TN"]

    total_pos = TP + FN
    total_neg = FP + TN
    total_pred_pos = TP + FP
    total_pred_neg = FN + TN
    total_all = total_pos + total_neg

    print("[Test set performance]")
    print("            Actual")
    print("         +            -   ")
    print(f"     +  {TP:<5}        {FP:<5}        {total_pred_pos:<5}")
    print("Pred ")
    print(f"     -  {FN:<5}        {TN:<5}        {total_pred_neg:<5}")
    print()
    print(f"        {total_pos:<5}        {total_neg:<5}        {total_all:<5}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        cm_values_path = os.path.join(outdir, f"confusion_matrix_values_{suffix}.txt")
        with open(cm_values_path, "w") as f:
            f.write("TP\tFP\tFN\tTN\n")
            f.write(f"{TP}\t{FP}\t{FN}\t{TN}\n")
        print(f"Confusion matrix values saved to: {cm_values_path}")

def plot_confusion_matrix_distillate(metrics: dict, outdir: str, model_type: str, dataset: str, suffix: str):
    """
    Plots confusion matrix with rows=Actual [Positive, Negative],
    columns=Predicted [Positive, Negative], matching the text printer.
    Uses dataset-specific class labels.
    """
    TP, FP, FN, TN = metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]

    if dataset.lower() == "mushroom":
        pos_label, neg_label = "poisonous", "edible"
    elif dataset.lower() == "adult":
        pos_label, neg_label = "gt_50k", "lte_50k"
    else:
        pos_label, neg_label = "Positive", "Negative"

    cm = [
        [TP, FN],  # Actual Positive -> (Pred Pos, Pred Neg)
        [FP, TN],  # Actual Negative -> (Pred Pos, Pred Neg)
    ]

    labels = [pos_label, neg_label]

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f'Confusion Matrix ({model_type.upper()}) for {dataset.capitalize() if dataset else "Dataset"}')

    os.makedirs(outdir, exist_ok=True)
    cm_path = os.path.join(outdir, f"confusion_matrix_{suffix}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to:", cm_path)


def load_y_true(folder: str, dataset: str) -> Tuple[Optional[str], Set[str]]:
    """
    Load ground-truth positives from {dataset}_y_true.f and negatives from {dataset}_y_true.n.
    Returns (predicate_name_or_None, set_of_positive_ids).
    """
    f_file = os.path.join(folder, f"{dataset}_y_true.f")
    n_file = os.path.join(folder, f"{dataset}_y_true.n")

    pos_ids: Set[str] = set()
    pred_name: Optional[str] = None

    # Load positives
    if os.path.exists(f_file):
        exs = load_examples(f_file)
        if exs:
            pred_name = exs[0][0]
            pos_ids |= {ex for _, ex in exs}

    # Load negatives only for universe, but we won't return them here
    neg_ids: Set[str] = set()
    if os.path.exists(n_file):
        exs = load_examples(n_file)
        if exs and pred_name is None:
            pred_name = exs[0][0]
        neg_ids |= {ex for _, ex in exs}


    return pred_name, pos_ids


def compute_true_confusion_for_folder(folder: str, dataset: str) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """
    True-label evaluation:
      - Load positives from {dataset}_y_true.pl.
      - Build the test universe from {dataset}_test.b if present
        (fallback to {dataset}.pl ONLY if _test.b is missing).
      - Predict via rules on that background.
      - Negatives are test IDs not listed as positive in y_test.

    Returns (cm_true_dict_or_None, acc_true_or_None)
    """
    rule_file   = os.path.join(folder, f"{dataset}_hypothesis.pl")
    bg_file = os.path.join(folder, f"{dataset}_test.b")

    # Required: rules and some background facts
    if not os.path.exists(rule_file):
        return None, None
    if os.path.exists(bg_file):
        b_file = bg_file
    else:
        return None, None

    # Ground-truth positives (y_test)
    y_pred_name, y_pos_ids = load_y_true(folder, dataset)
    if not y_pos_ids:
        return None, None
    
    # Load rules + background facts
    rules = parse_rules(rule_file)
    features = load_features(b_file)

    # Test universe = all example IDs present in the chosen background facts
    test_ids = set(features.keys())
    if not test_ids:
        return None, None

    # Choose target predicate for prediction
    target_pred = y_pred_name or (rules[0][0] if rules else None)
    if target_pred is None:
        return None, None

    # Compute confusion vs true labels
    TP = FP = FN = TN = 0
    for ex in test_ids:
        y_true = (ex in y_pos_ids)
        y_hat  = predicts(rules, ex, features, target_pred)
        if y_true and y_hat:
            TP += 1
        elif (not y_true) and y_hat:
            FP += 1
        elif y_true and (not y_hat):
            FN += 1
        else:
            TN += 1

    total = TP + FP + FN + TN
    acc_true = (TP + TN) / total if total else 0.0
    cm_true = {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "Accuracy": acc_true}
    return cm_true, acc_true

def compute_compactness(folder: str, dataset: str) -> Dict[str, Union[int, float]]:
    """
    Compactness: number of target rules and their average body length.
    """
    rule_file = os.path.join(folder, f"{dataset}_hypothesis.pl")
    rules = parse_rules(rule_file)

    # Determine target predicate name
    target_pred = rules[0][0] if rules else None

    # Compactness computed for rules of the target predicate
    target_rules = [(h, conds) for (h, conds) in rules if (target_pred is None or h == target_pred)]
    num_rules = len(target_rules)
    avg_body_len = (mean(len(conds) for _, conds in target_rules) if target_rules else 0.0)
    comp = {"num_rules": num_rules, "avg_body_len": avg_body_len}
    return comp


def evaluate_folder(folder: str, dataset: str, model_type: str) -> Dict[str, Union[float, Dict]]:
    """
    Orchestrates the full evaluation and returns the requested dictionary.
    - confusion: fidelity confusion matrix vs {_test.f/.n}
    - fidelity: accuracy from that confusion
    - accuracy_true: accuracy vs ground-truth {_y_true.pl} (or None if unavailable)
    - coverage, compactness
    """
    # Fidelity confusion (already implemented in your code)
    cm_fid = compute_fidelity_confusion_for_folder(folder, dataset)
    # Plot and print confusion matrix
    print_confusion_matrix_distillate(cm_fid, outdir=folder, suffix="wrt_model_predictions")
    plot_confusion_matrix_distillate(cm_fid, outdir=folder, model_type=model_type, dataset=dataset, suffix="wrt_model_predictions")
    # Fidelity accuracy
    fidelity = cm_fid.get("Accuracy", None)

    # Ground-truth accuracy
    cm_true, acc_true = compute_true_confusion_for_folder(folder, dataset)
    # Plot and print true confusion matrix if available
    if cm_true:
        print_confusion_matrix_distillate(cm_true, outdir=folder, suffix="wrt_true_labels")
        plot_confusion_matrix_distillate(cm_true, outdir=folder, model_type=model_type, dataset=dataset, suffix="wrt_true_labels")

    comp = compute_compactness(folder, dataset)

    # Compute coverage as recall on the fidelity confusion
    TP = cm_fid.get("TP", 0)
    FN = cm_fid.get("FN", 0)
    cov = {"value": (TP / (TP + FN) if (TP + FN) > 0 else 0.0), "covered": TP, "total": (TP + FN)}

    # Save the evaluation results to a file in the current folder
    eval_results = {
        "accuracy_true": acc_true,            # may be None if y_test missing
        "fidelity": fidelity,                 # equals cm_fid["Accuracy"]
        "coverage": cov,                      # {"value", "covered", "total"}
        "compactness": comp,                  # {"num_rules", "avg_body_len"}
    }

    eval_results_path = os.path.join(folder, "evaluation_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print("Evaluation results saved to:", eval_results_path)

    return eval_results
