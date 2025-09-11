#!/usr/bin/env python3
# scripts/train_and_export_aleph_single.py

import os, re, joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

def _to_atom(s: str) -> str:
    """
    Converts a given string to a valid atom name by replacing non-alphanumeric characters with underscores,
    ensuring it starts with a letter or underscore, and converting it to lowercase.

    Args:
        s (str): The input string to be converted.

    Returns:
        str: A valid atom name in lowercase, with non-alphanumeric characters replaced and a leading 'v_' if necessary.
    """
    s = str(s)
    s = re.sub(r'[^A-Za-z0-9_]', '_', s)
    if not re.match(r'^[A-Za-z_]', s):
        s = 'v_' + s
    return s.lower()

def train_and_export_aleph_single(model_type="dt", dataset="mushroom"):
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset}.parquet")

    if os.path.exists(cache_path):
        print(f"Loading cached {dataset.capitalize()} dataset from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"Downloading {dataset.capitalize()} dataset...")
        version = 1 if dataset == "mushroom" else 2 if dataset == "adult" else None
        if version is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        data = fetch_openml(name=dataset, version=version, as_frame=True)
        df = data.frame
        if dataset == "adult":
            # Remove rows with missing values (marked as '?')
            df = df.replace('?', pd.NA).dropna()
            # The target column is 'class' in mushroom, but 'income' in adult
            df = df.rename(columns={'income': 'class'})
        df.to_parquet(cache_path)
        print(f"Cached {dataset.capitalize()} dataset at {cache_path}.")

    # Split
    X = df.drop('class', axis=1)     # categorical
    y = df['class']                  # 'e' or 'p'

    # Keep original X for ILP; encode only for sklearn
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    X_enc_df = pd.DataFrame(X_enc, index=X.index, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc_df, y, test_size=0.3, random_state=42, stratify=y
    )

    # Encode y for XGBoost (expects numeric labels)
    if model_type == "xgb":
        y_le = LabelEncoder()
        y_train = y_le.fit_transform(y_train)
        y_test = y_le.transform(y_test)

    # --- Model zoo ---
    models = {
        "dt": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [2, 4, 6, 8, 10],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        ),
        "rf": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5],
                "criterion": ["gini", "entropy"],
            },
        ),
        "xgb": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        ),
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {list(models.keys())}.")

    base_model, param_grid = models[model_type]

    # Grid search
    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_

    y_pred = clf.predict(X_test)
    print(f"Model: {model_type.upper()}")
    print("Best params:", grid.best_params_)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Output paths
    outdir = os.path.join(os.path.dirname(__file__), '..', 'outputs', model_type)
    os.makedirs(outdir, exist_ok=True)
    pl_path = os.path.join(outdir, 'pred_edible.pl')

    # Stable IDs for test set
    test_ids = [f"m{i+1}" for i in range(len(X_test))]
    id_map = pd.Series(test_ids, index=X_test.index)

    # Build sections
    # Aleph header for the generated Prolog file.
    # The following lines configure Aleph's behavior and search parameters.
    clause_length = 5
    nodes = 50000
    noise = 100
    verbose = 0 # 0=silent, 1=normal, 2=verbose
    header = [
        ":- use_module(library(aleph)).",  # Load Aleph library
        ":- aleph.",                       # Initialize Aleph
        f":- aleph_set(verbose, {verbose}).",      # (Optional) Set verbosity: 0=silent, 1=normal, 2=verbose
        f":- aleph_set(clause_length, {clause_length}).", # Set max clause length
        "%:- aleph_set(i, 4).",            # https://www.swi-prolog.org/pack/file_details/aleph/doc/manual.html#manual
        f":- aleph_set(nodes, {nodes}).",  # Set max number of nodes explored
        f":- aleph_set(noise, {noise}).",  # Allow up to {noise} negative examples per clause
        "% The following modes and determination declare the learning bias",
        ":- modeh(*, pred_edible(+id)).",  # Head mode: target predicate
        ":- modeb(*, feature(+id, #feature, #value)).", # Body mode: features
        ":- determination(pred_edible/1, feature/3).",  # Allow features in rules for pred_edible
        ""
    ]

    # Background facts with ORIGINAL (unencoded) X
    bg = [":- begin_bg."]
    for orig_idx in X_test.index:
        mid = id_map[orig_idx]
        row = X.loc[orig_idx]
        for col, val in row.items():
            ftr = _to_atom(col)
            v = _to_atom(val)
            bg.append(f"feature({mid}, {ftr}, {v}).")
    bg.append(":- end_bg.\n")

    # Positives/negatives from MODEL predictions
    y_pred_series = pd.Series(y_pred, index=X_test.index)
    pos = [":- begin_in_pos."]
    neg = [":- begin_in_neg."]
    for orig_idx, pred in y_pred_series.items():
        mid = id_map[orig_idx]
        # For XGBoost, predictions are numeric, so map back to original labels
        if model_type == "xgb":
            label = y_le.inverse_transform([pred])[0]
        else:
            label = pred
        if label == 'e':
            pos.append(f"pred_edible({mid}).")
        else:
            neg.append(f"pred_edible({mid}).")
    pos.append(":- end_in_pos.\n")
    neg.append(":- end_in_neg.\n")

    # Write Aleph program
    with open(pl_path, "w") as f:
        f.write("\n".join(header + bg + pos + neg))

    print("Aleph program written to:", pl_path)

    # Save best hyperparameters to a file
    params_path = os.path.join(outdir, 'best_params.txt')
    with open(params_path, 'w') as f:
        for k, v in grid.best_params_.items():
            f.write(f"{k}: {v}\n")
    print("Best hyperparameters written to:", params_path)

