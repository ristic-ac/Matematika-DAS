import os
import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

Feature = Tuple[str, str, str]
Rule = Tuple[str, List[Feature]]

def parse_rules(rule_file: str) -> List[Rule]:
    rules: List[Rule] = []
    head_pat = re.compile(r"^(\w+)\s*\(\s*\w+\s*\)\s*:-\s*(.*)\.$")
    cond_pat = re.compile(r"(\w+)\s*\(\s*\w+\s*,\s*([^,]+)\s*,\s*([^\)]+)\s*\)")
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
            conds = [(p.strip(), a.strip(), v.strip()) 
                     for (p, a, v) in cond_pat.findall(body)]
            rules.append((target, conds))
    return rules

def load_features(b_file: str) -> Dict[str, Dict[Tuple[str,str], str]]:
    feats: Dict[str, Dict[Tuple[str,str], str]] = {}
    pat = re.compile(r"(\w+)\s*\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^\)]+)\s*\)\s*\.")
    with open(b_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            m = pat.match(line)
            if not m:
                continue
            pred, ex, attr, val = m.groups()
            feats.setdefault(ex, {})[(pred.strip(), attr.strip())] = val.strip()
    return feats

def load_examples(f_file: str) -> List[Tuple[str, str]]:
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

def predicts(rules: List[Rule], ex_id: str, features: Dict[str, Dict[Tuple[str,str], str]], target_pred: str) -> bool:
    if ex_id not in features:
        return False
    fdict = features[ex_id]
    for (head, conds) in rules:
        if head != target_pred:
            continue
        if all((pred, attr) in fdict and fdict[(pred, attr)] == val for pred, attr, val in conds):
            return True
    return False

def compute_confusion_for_folder(folder: str, dataset: str) -> Dict[str, float]:
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

def print_confusion_matrix_distillate(metrics: Dict[str, int]):
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



def plot_confusion_matrix_distillate(metrics: dict, outdir: str, model_type: str, dataset: str):
    """
    Plots confusion matrix with rows=Actual [Positive, Negative],
    columns=Predicted [Positive, Negative], matching the text printer.
    Uses dataset-specific class labels.
    """
    TP, FP, FN, TN = metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]

    # Dataset-specific labels
    if dataset.lower() == "mushroom":
        pos_label, neg_label = "edible", "poisonous"
    elif dataset.lower() == "adult":
        pos_label, neg_label = "gt_50k", "lte_50k"
    else:
        pos_label, neg_label = "Positive", "Negative"

    # Correct layout: rows = Actual (P,N), cols = Predicted (P,N)
    cm = [
        [TP, FN],  # Actual Positive -> (Pred Pos = TP, Pred Neg = FN)
        [FP, TN],  # Actual Negative -> (Pred Pos = FP, Pred Neg = TN)
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
    cm_path = os.path.join(outdir, "confusion_matrix_distillate.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to:", cm_path)

