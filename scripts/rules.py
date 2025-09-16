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
        if len(parts) > expected_parts:
            # merge extras into last
            head = parts[:expected_parts-1]
            tail = [", ".join(parts[expected_parts-1:])]
            parts = head + tail
        else:
            parts = parts + [""] * (expected_parts - len(parts))
    return parts

def _normalize_head(head_str: str) -> str:
    """Normalize clause head to keep predicate name but replace arg with A."""
    head_str = head_str.strip()
    if "(" in head_str and head_str.endswith(")"):
        fun = head_str.split("(", 1)[0].strip().strip("'")
        return f"{fun}(A)"
    return "pred_unknown(A)"  # fallback

def _parse_feature_literal(lit: str) -> str:
    """
    Normalize any predicate literal so its first argument becomes A.
      color(X, red)     -> color(A, red)
      shape(Y, circle)  -> shape(A, circle)
      p(Z)              -> p(A)
    Leaves non-call atoms (e.g., 'true') unchanged.
    """
    s = _strip_enclosing_parens(lit.strip())
    if "(" not in s or not s.endswith(")"):
        return s  # not a predicate call, pass through

    fun = s.split("(", 1)[0].strip().strip("'")
    inside, _ = _extract_paren_content(s, s.find("("))
    first, rest = _split_top_level_once(inside)

    args = ["A"]
    if rest is not None and rest.strip() != "":
        args.append(rest.strip())

    return f"{fun}({', '.join(args)})"


def _flatten_body(body_str: str):
    s = _strip_enclosing_parens(body_str.strip())

    if s.startswith(",("):
        inside, _ = _extract_paren_content(s, 1)
        left, right = _split_top_level_once(inside)
        return _flatten_body(left) + _flatten_body(right)

    left, right = _split_top_level_once(s)
    if right is not None:
        return _flatten_body(left) + _flatten_body(right)

    s = _strip_enclosing_parens(s)
    if s == "true" or s == "":
        return []
    return [_parse_feature_literal(s)]


def clean_aleph_clause(clause_str: str) -> str | None:
    """
    Transform Aleph clause term string into cleaned Prolog-style rule:
      pred(A) :- feat(A, val), other_feat(A, val).
    Returns None if it's just a fact (e.g., pred(A).).
    """
    s = _strip_quotes_and_period(clause_str)

    s_ = s[1:] if s.startswith("'") else s
    if s_.startswith(":-"):
        idx = s_.find("(")
        if idx == -1:
            return None  # malformed, treat as skipped
        inner, _ = _extract_paren_content(s_, idx)
        head_str, body_str = _split_top_level_once(inner)
        head = _normalize_head(head_str)
        body_lits = _flatten_body(body_str) if body_str else []
        if body_lits:
            return f"{head} :- {', '.join(body_lits)}."
        else:
            return None  # skip head-only rule
    else:
        # fact-only clause — skip
        return None

def clean_aleph_program(program_terms, out_path=None):
    """
    Keep only *rules* (no facts).
    """
    cleaned = []
    for t in program_terms:
        clause = clean_aleph_clause(str(t))
        if clause is not None:
            cleaned.append(clause)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for line in cleaned:
                f.write(line + "\n")
    return cleaned
