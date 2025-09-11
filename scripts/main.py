import sys
from pyswip import Prolog
import os

from train_and_export import train_and_export_aleph_single


import os

import os

def _strip_quotes_and_period(s: str) -> str:
    s = s.strip()
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    if s.endswith("."):
        s = s[:-1]
    return s.strip()

def _extract_paren_content(s: str, open_idx: int):
    """Return (inside, close_idx) for the '(' at open_idx."""
    assert s[open_idx] == "(", "expected '('"
    depth = 0
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i], i
    raise ValueError("Unbalanced parentheses")

def _strip_enclosing_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        inner, close = _extract_paren_content(s, 0)
        if close == len(s) - 1:
            s = inner.strip()
        else:
            break
    return s

def _split_top_level_once(s: str):
    """Split on the first top-level comma (not inside parens)."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i].strip(), s[i+1:].strip()
    return s.strip(), None

def _split_top_level_commas(s: str, expected_parts: int):
    """Split into expected_parts by top-level commas."""
    parts = []
    depth = 0
    buf = []
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    if len(parts) != expected_parts:
        # Fallback: pad or merge to reach expected_parts
        if len(parts) > expected_parts:
            # merge extras into last
            head = parts[:expected_parts-1]
            tail = [", ".join(parts[expected_parts-1:])]
            parts = head + tail
        else:
            parts = parts + [""] * (expected_parts - len(parts))
    return parts

def _normalize_head(head_str: str) -> str:
    head_str = head_str.strip()
    # Expect pred_edible(<var>)
    if head_str.startswith("pred_edible("):
        return "pred_edible(A)"
    # If it’s already quoted functor or something odd, do a best effort
    # Try to get content inside the first '('
    if "(" in head_str and head_str.endswith(")"):
        fun = head_str.split("(", 1)[0].strip().strip("'")
        return f"{fun}(A)"
    return "pred_edible(A)"  # safe default

def _parse_feature_literal(lit: str) -> str:
    # Expect feature(<var>, <feat>, <val>)
    lit = lit.strip()
    if not lit.startswith("feature("):
        return lit  # leave unknown literals untouched
    inside, _ = _extract_paren_content(lit, lit.find("("))
    a0, a1, a2 = _split_top_level_commas(inside, 3)
    a1, a2 = a1.strip(), a2.strip()
    return f"feature(A, {a1}, {a2})"

def _flatten_body(body_str: str):
    """
    Return a flat list of literal strings from Aleph’s nested conjunctions.
    Handles forms like:
      ",(L1, L2)", "L1, ,(L2, L3)", "(L1, L2)", and single 'feature(...)'.
    """
    s = _strip_enclosing_parens(body_str.strip())

    # Case 1: explicit conjunction functor ",( ... )"
    if s.startswith(",("):
        inside, _ = _extract_paren_content(s, 1)  # index of '(' after ','
        left, right = _split_top_level_once(inside)
        return _flatten_body(left) + _flatten_body(right)

    # Case 2: top-level comma between two chunks (e.g., "feature(...), ,( ... )")
    left, right = _split_top_level_once(s)
    if right is not None and not s.startswith("feature("):
        return _flatten_body(left) + _flatten_body(right)

    # Case 3: single literal or parenthesized literal
    s = _strip_enclosing_parens(s)
    if s == "true" or s == "":
        return []
    return [_parse_feature_literal(s)]

def clean_aleph_clause(clause_str: str) -> str:
    """
    Transform Aleph clause term string into:
      pred_edible(A) :- feature(A, feat, val), feature(A, feat, val).
    Or a fact:
      pred_edible(A).
    """
    s = _strip_quotes_and_period(clause_str)

    # Clause of the form ':- (Head, Body)' or "':-(Head, Body)"
    # Accept both ":-(" and "':-(" prefixes
    s_ = s[1:] if s.startswith("'") else s
    if s_.startswith(":-"):
        # find first '(' after ':-'
        idx = s_.find("(")
        if idx == -1:
            # malformed; fall back
            return "pred_edible(A)."
        inner, _ = _extract_paren_content(s_, idx)
        head_str, body_str = _split_top_level_once(inner)
        head = _normalize_head(head_str)
        body_lits = _flatten_body(body_str) if body_str else []
        if body_lits:
            return f"{head} :- {', '.join(body_lits)}."
        else:
            return f"{head}."
    else:
        # Fact like "pred_edible(_2044)" (no body)
        head = _normalize_head(s)
        return f"{head}."

def clean_aleph_program(program_terms, out_path=None):
    """
    program_terms: iterable of clause terms from induce(Program) (strings/objects)
    out_path: optional file to write to
    returns list of cleaned clause strings
    """
    cleaned = [clean_aleph_clause(str(t)) for t in program_terms]
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for line in cleaned:
                f.write(line + "\n")
    return cleaned

def run_aleph_with_files(model_type: str):
    """
    Run Aleph by consulting only pred_edible.pl in the corresponding model_type folder, as in swi.sh.
    """
    # Build paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "outputs", model_type)
    pred_file = os.path.join(base_dir, "pred_edible.pl")

    prolog = Prolog()

    # Load Aleph
    list(prolog.query("use_module(library(aleph))"))

    print(f"[Aleph] Consulting: {pred_file}")
    list(prolog.query(f"consult('{pred_file}')"))

    res = list(prolog.query("induce(Program)"))
    if res:
        program = res[0]['Program']
        cleaned = clean_aleph_program(program, out_path=os.path.join(base_dir, "hypothesis.pl"))
        for c in cleaned:
            print(c)
    else:
        print("No hypothesis term.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("No arguments provided, defaulting to: both dt")
        action = "both"
        model_arg = "dt"
    else:
        model_arg = sys.argv[1].lower()
        action = sys.argv[2].lower()

    if model_arg == "all":
        models = ["dt", "rf", "xgb"]
    elif model_arg in ["dt", "rf", "xgb"]:
        models = [model_arg]
    else:
        print("Invalid model argument. Use one of: dt, rf, xgb, all")
        sys.exit(1)

    for model in models:
        if action == "train":
            train_and_export_aleph_single(model_type=model)
        elif action == "aleph":
            run_aleph_with_files(model_type=model)
        elif action == "both":
            train_and_export_aleph_single(model_type=model)
            run_aleph_with_files(model_type=model)
        else:
            print("Invalid action. Use one of: train, aleph, both")
            sys.exit(1)
