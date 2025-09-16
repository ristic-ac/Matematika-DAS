#!/usr/bin/env python3
# scripts/train_and_export_aleph_single.py

import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
    ),
}

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

def save_feature_importances(clf, X, outdir, model_type):
    """
    Save and print feature importances for supported models.
    """
    if model_type in ["dt", "rf", "xgb"]:
        importances = clf.feature_importances_
        feature_names = X.columns
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("Feature importances:")
        print(feat_imp)
        # Optionally, save to file
        imp_path = os.path.join(outdir, 'feature_importances.csv')
        feat_imp.to_csv(imp_path, header=True)
        print("Feature importances saved to:", imp_path)

def save_and_print_confusion_matrix_model(clf, y_test, y_pred, outdir, model_type, dataset=None):
    y_pred = [int(x) for x in y_pred]
    y_test = [int(x) for x in y_test]

    if dataset == "mushroom":
        labels = [0, 1]
        custom_labels = ["poisonous", "edible"]
    elif dataset == "adult":
        labels = [0, 1]
        custom_labels = ["lte_50K", "gt_50K"]
    else:
        labels = sorted(set(y_test))
        custom_labels = [str(c) for c in (getattr(clf, "classes_", labels))]

    label_map = dict(zip(labels, custom_labels))
    y_pred_labels = [label_map[x] for x in y_pred]
    y_test_labels = [label_map[x] for x in y_test]

    # Switch order of custom_labels to swap TP and TN
    custom_labels_swapped = custom_labels[::-1]

    cm_raw = confusion_matrix(y_test_labels, y_pred_labels, labels=custom_labels_swapped)
    df_cm = pd.DataFrame(cm_raw, index=custom_labels_swapped, columns=custom_labels_swapped)

    print("Confusion Matrix:")
    print(df_cm)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix ({model_type.upper()}) for {dataset.capitalize() if dataset else "Dataset"}')

    os.makedirs(outdir, exist_ok=True)
    cm_path = os.path.join(outdir, 'confusion_matrix_model.png')
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to:", cm_path)

def plot_feature_correlation_heatmap(X, dataset):
    """
    Plots and saves the correlation heatmap for encoded features.
    """
    plt.figure(figsize=(12, 10))
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    X_enc_df = pd.DataFrame(X_enc, index=X.index, columns=X.columns)
    corr = X_enc_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(f'Feature Correlation Heatmap ({dataset.capitalize()})')
    heatmap_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', dataset, f'{dataset}_feature_correlation_heatmap.png')
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    plt.savefig(heatmap_path)
    plt.close()
    print("Feature correlation heatmap saved to:", heatmap_path)

def train_and_export_aleph_single(model_type, dataset, aleph_settings):
    aleph_folder, aleph_hyperparams = list(aleph_settings)
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
            df = df.rename(columns={'income': 'class'})
        df.to_parquet(cache_path)
        print(f"Cached {dataset.capitalize()} dataset at {cache_path}.")

    
    # If mushroom dataset, rename 'bruises%3f' to 'bruises' and fix values
    if dataset == "mushroom":
        df = preprocess_mushroom_dataset(df)

    # Split features and target
    X = df.drop('class', axis=1)
    y = df['class']

    # Drop rows with missing values
    if X.isnull().any().any():
        unique_rows_before = df.shape[0]
        print(f"Dataset has {unique_rows_before} rows before dropping missing values.")
        print("Ratio of missing data per feature:")
        print(X.isnull().mean())
        # Plot the correlation between stalk-root and class for mushroom dataset
        if dataset == "mushroom":
            if pd.api.types.is_categorical_dtype(df['stalk-root']):
                df['stalk-root'] = df['stalk-root'].cat.add_categories(['unknown'])
            df['stalk-root'] = df['stalk-root'].fillna('unknown')
            # Create contigency table
            cont_table = pd.crosstab(df['stalk-root'], df['class'], dropna=False)
            cont_table_norm = cont_table.div(cont_table.sum(axis=1), axis=0)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cont_table_norm, annot=True, fmt=".2f", cmap='Blues')
            plt.title('Normalized Contingency Table: stalk-root vs class')
            plt.xlabel('class')
            plt.ylabel('stalk-root')
            heatmap_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', dataset, 'charts' , 'categorical', f'{dataset}_stalk_root_vs_class_heatmap.png')
            os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
            plt.savefig(heatmap_path)
            plt.close()
            print("Contingency table heatmap saved to:", heatmap_path)
        missing_rows = df[df.isnull().any(axis=1)]
        num_missing = missing_rows.shape[0]
        print(f"Missing values detected. {num_missing} unique rows will be dropped due to missing values.")
        df = df.dropna()
        X = df.drop('class', axis=1)
        y = df['class']

    # Describe dataset
    print(f"Dataset: {dataset.capitalize()}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {X.shape[1]}")
    print("Feature types:")
    print(X.dtypes)
    print("Summary statistics:")
    print(X.describe(include='all'))
    print("Target class distribution:")
    print(y.value_counts())

    # Check output class distribution
    print("Class distribution:")
    print(y.value_counts())

    # Describe numerical fields
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print("Numerical feature statistics:")
        print(X[num_cols].describe())
    else:
        print("No numerical features found.")

    charts_outdir = os.path.join(os.path.dirname(__file__), '..', 'outputs', dataset, 'charts')
    numerical_outdir = os.path.join(charts_outdir, 'numerical')
    categorical_outdir = os.path.join(charts_outdir, 'categorical')

    generate_overlapping_histograms(X, y, numerical_outdir, dataset)
    generate_categorical_barcharts(X, y, categorical_outdir, dataset)

    # Get whichever class is minority
    minority_class = y.value_counts().idxmin()
    minority_count = y.value_counts().min()
    print(f"Minority class: {minority_class} with {minority_count} samples.")
    # Make sure equal size classes
    if y.value_counts().nunique() > 1:
        print("Balancing classes by downsampling majority class...")
        df_minority = df[y == minority_class]
        df_majority = df[y != minority_class].sample(n=minority_count, random_state=42)
        df = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=42).reset_index(drop=True)
        X = df.drop('class', axis=1)
        y = df['class']
        print("New class distribution:")
        print(y.value_counts())
    else:
        print("Classes are already balanced.")
    
    # Print class distribution after balancing
    print("Final class distribution:")
    print(y.value_counts())

    # Custom binning for Adult dataset
    if dataset == "adult":
        print("Applying custom binning to Adult dataset...")
        X = bin_adult_dataset(X)
        print("Binning applied. New feature types:")
        print(X.dtypes)
        print("Summary statistics after binning:")
        print(X.describe(include='all'))

    # If feature_importances.csv exists, load and keep only top 90%
    outdir = os.path.join(os.path.dirname(__file__), '..', 'outputs', dataset, model_type)
    imp_path = os.path.join(outdir, 'feature_importances.csv')
    if os.path.exists(imp_path):
        print(f"Loading feature importances from {imp_path} to select top features...")
        feat_imp = pd.read_csv(imp_path, index_col=0)
        if feat_imp.shape[1] == 1:
            feat_imp = feat_imp.iloc[:, 0]
        feat_imp = feat_imp.sort_values(ascending=False)
        cum_importance = feat_imp.cumsum() / feat_imp.sum()
        top_features = cum_importance[cum_importance < 0.9].index.tolist()
        if len(top_features) < len(feat_imp):
            # Add the next feature to ensure >=90% coverage
            next_feature = cum_importance.index[len(top_features)]
            top_features.append(next_feature)
        print("Top features and their importances:")
        print(feat_imp.loc[top_features])
        print(f"Selected top {len(top_features)} features covering at least 90% importance.")
        print("Total sum of importances for selected features:", feat_imp.loc[top_features].sum())
        X = X[top_features]

    # Keep original X for ILP; encode only for sklearn
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    X_enc_df = pd.DataFrame(X_enc, index=X.index, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc_df, y, test_size=0.15, random_state=3, stratify=y
    )

    # Print number of samples in train and test sets
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Encode y. For mushrooms p=0, e=1; for adult <=50K=0, >50K=1
    # We do it manually for each model
    if dataset == "mushroom":
        y_train = y_train.map({'poisonous': 0, 'edible': 1}).astype(int)
        y_test = y_test.map({'poisonous': 0, 'edible': 1}).astype(int)
    elif dataset == "adult":
        y_train = y_train.map({'<=50K': 0, '>50K': 1}).astype(int)
        y_test = y_test.map({'<=50K': 0, '>50K': 1}).astype(int)

    if model_type not in models:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {list(models.keys())}.")

    base_model, param_grid = models[model_type]

    # Output paths (needed for params_path)
    outdir = os.path.join(os.path.dirname(__file__), '..', 'outputs', dataset, model_type)
    os.makedirs(outdir, exist_ok=True)
    params_path = os.path.join(outdir, 'best_params.txt')

    best_params = load_best_params(params_path)
    if best_params:
        print(f"Found best_params.txt, using parameters: {best_params}")
        clf = base_model.set_params(**best_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        grid = type('DummyGrid', (), {'best_params_': best_params})()  # Dummy for later use
    else:
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
        # Save best hyperparameters to a file
        params_path = os.path.join(outdir, 'best_params.txt')
        with open(params_path, 'w') as f:
            for k, v in grid.best_params_.items():
                f.write(f"{k}: {v}\n")
        print("Best hyperparameters written to:", params_path)
        # Print message to re-run for final model
        print("Re-run the script to train final model with best hyperparameters.")
        exit(0)

    # Output paths
    outdir = os.path.join(os.path.dirname(__file__), '..', 'outputs', dataset, model_type)
    os.makedirs(outdir, exist_ok=True)

    # Save which columns were used for training
    cols_path = os.path.join(outdir, 'feature_columns.txt')
    with open(cols_path, 'w') as f:
        f.write("\n".join(X_train.columns))
    print("Feature columns written to:", cols_path)

    save_and_print_confusion_matrix_model(
        clf, y_test, y_pred, outdir, model_type, dataset
    )

    # Only save feature importances if the file does not already exist
    if not os.path.exists(os.path.join(outdir, 'feature_importances.csv')):
        save_feature_importances(clf, X, outdir, model_type)
        print("Feature importances computed and saved. Exiting to allow re-run with top features only.")
        sys.exit(0)

    # Append aleph folder to outdir for Aleph files
    outdir = os.path.join(outdir, aleph_folder)
    os.makedirs(outdir, exist_ok=True)
    print("Aleph output directory:", outdir)
    
    # Stable IDs for all examples (train + test)
    all_ids = [f"m{i+1}" for i in range(len(X))]
    id_map = pd.Series(all_ids, index=X.index)

    pred_label = 'edible' if dataset == "mushroom" else 'gt_50K'

    # Build complete background knowledge (training + testing)
    bg = [":- begin_bg."]
    for orig_idx in X.index:  # X_full = training + testing data
        mid = f"m{orig_idx+1}" if isinstance(orig_idx, int) else f"m{X.index.get_loc(orig_idx)+1}"
        row = X.loc[orig_idx]
        for col, val in row.items():
            ftr = _to_atom(col)
            v = _to_atom(val)
            bg.append(f"feature({mid}, {ftr}, {v}).")
    bg.append(":- end_bg.\n")

    # Positives/negatives from TRAINING labels
    pos = [":- begin_in_pos."]
    neg = [":- begin_in_neg."]
    for orig_idx, label in y_train.items():
        mid = f"m{orig_idx+1}" if isinstance(orig_idx, int) else f"m{X.index.get_loc(orig_idx)+1}"
        if label == 1:
            pos.append(f"{pred_label}({mid}).")
        else:
            neg.append(f"{pred_label}({mid}).")
    pos.append(":- end_in_pos.\n")
    neg.append(":- end_in_neg.\n")

    # Write Aleph training program with full BG + training pos/neg
    pl_path = os.path.join(outdir, f"{dataset}.pl")
    with open(pl_path, "w") as f:
        aleph_preamble = generate_aleph_header_lines()
        aleph_modes = generate_aleph_modes(pred_label)
        f.write("\n".join([aleph_preamble] + [aleph_hyperparams] + [aleph_modes] + bg + pos + neg))
    print("Aleph training program written to:", pl_path)

    # Write positive test examples into test.f
    test_f_path = os.path.join(outdir, f'{dataset}_test.f')
    with open(test_f_path, "w") as f:
        f.write("\n".join([f"{pred_label}({id_map[idx]})." for idx, label in y_test.items() if label == 1]))
    print("Positive test examples written to:", test_f_path)

    # Write negative test examples into test.n
    test_n_path = os.path.join(outdir, f'{dataset}_test.n')
    with open(test_n_path, "w") as f:
        f.write("\n".join([f"{pred_label}({id_map[idx]})." for idx, label in y_test.items() if label == 0]))
    print("Negative test examples written to:", test_n_path)

    # Write test background knowledge into test.b
    test_b_path = os.path.join(outdir, f'{dataset}_test.b')
    with open(test_b_path, "w") as f:
        for orig_idx in y_test.index:
            mid = f"m{orig_idx+1}" if isinstance(orig_idx, int) else f"m{X.index.get_loc(orig_idx)+1}"
            row = X.loc[orig_idx]
            for col, val in row.items():
                ftr = _to_atom(col)
                v = _to_atom(val)
                f.write(f"feature({mid}, {ftr}, {v}).\n")
    print("Test background knowledge written to:", test_b_path)


# Function that creates histograms for numerical features in the outdir
# Name should include dataset name for clarity (e.g., adult_age_histogram.png)
# X represents the inputs DataFrame, y the target Series
# Outdir represents the output directory for saving plots
def generate_overlapping_histograms(X, y, outdir, dataset_name):
    # Identify numerical columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        print("No numerical columns to plot.")
        return

    # Create overlapping histograms for each numerical column
    for col in num_cols:
        plot_path = os.path.join(outdir, f'{dataset_name}_{col}_overlap_histogram.png')
        if os.path.exists(plot_path):
            print(f"Overlapping histogram for {col} already exists at {plot_path}, skipping.")
            continue
        plt.figure(figsize=(8, 6))
        # Compute common bins for all classes
        data_all = X[col].dropna()
        bins = np.histogram_bin_edges(data_all, bins=30)
        for class_value in sorted(y.unique()):
            sns.histplot(
                X[y == class_value][col],
                bins=bins,
                kde=False,
                stat="density",
                label=str(class_value),
                alpha=0.5
            )
        plt.title(f'Overlapping Histogram of {col} ({dataset_name.capitalize()})')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(title='Class')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved overlapping histogram for {col} to {plot_path}")

def generate_categorical_barcharts(X, y, outdir, dataset_name, log_threshold=50):
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        print("No categorical columns to plot.")
        return

    for col in cat_cols:
        plot_path = os.path.join(outdir, f'{dataset_name}_{col}_barchart.png')
        if os.path.exists(plot_path):
            print(f"Bar chart for {col} already exists at {plot_path}, skipping.")
            continue

        plt.figure(figsize=(10, 6))
        plot_df = pd.concat([X[col], y], axis=1)
        plot_df.columns = [col, 'target']
        
        # Count values to decide on log scale
        counts = plot_df.groupby([col, 'target'], observed=True).size()
        if len(counts) > 0:
            max_count = counts.max()
            min_count = counts.min() if counts.min() > 0 else 1  # avoid divide by zero
            use_log = (max_count / min_count) > log_threshold
        else:
            use_log = False
        
        # Plot
        ax = sns.countplot(data=plot_df, x=col, hue='target', dodge=True)
        plt.title(f'Bar Chart of {col} by Class ({dataset_name.capitalize()})')
        plt.xlabel(col)
        plt.ylabel('Count (log scale)' if use_log else 'Count')
        plt.legend(title='Class')
        plt.xticks(rotation=45, ha='right')
        
        # Apply log scale if needed
        if use_log:
            ax.set_yscale('log')

        # Save plot
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar chart for {col} to {plot_path} (log scale: {use_log})")


def load_best_params(params_path):
    """
    Load best hyperparameters from a text file.

    The file is expected to contain lines of the form:
        key: value

    Recognizes “none”, “true”, “false” (case insensitive) and tries to convert
    numeric strings to int or float.

    Args:
        params_path (str): Path to the params file.

    Returns:
        dict or None: Dictionary of parameters if file exists and is parseable, else None.
    """
    if not os.path.exists(params_path):
        return None

    best_params = {}
    with open(params_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip()
            v = v.strip()

            # Normalize value
            v_lower = v.lower()
            if v_lower == "none":
                parsed_v = None
            elif v_lower == "true":
                parsed_v = True
            elif v_lower == "false":
                parsed_v = False
            else:
                # Try int
                try:
                    parsed_v = int(v)
                except ValueError:
                    # Try float
                    try:
                        parsed_v = float(v)
                    except ValueError:
                        # Leave as string
                        parsed_v = v
            best_params[k] = parsed_v

    return best_params

import pandas as pd
import numpy as np

def bin_adult_dataset(X):
    # Age bins
    age_bins = [0, 25, 35, 45, 55, 65, 150]
    age_labels = ["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
    X["age"] = pd.cut(X["age"].astype(float), bins=age_bins, labels=age_labels, right=False)

    # Hours-per-week bins
    hpw_bins = [0, 21, 41, 61, 1000]
    hpw_labels = ["≤20", "21-40", "41-60", ">60"]
    X["hours-per-week"] = pd.cut(X["hours-per-week"].astype(float), bins=hpw_bins, labels=hpw_labels, right=False)

    # Capital-gain bins using quartiles (non-zero values only)
    cg = X["capital-gain"].astype(float)
    cg_q25 = cg[cg > 0].quantile(0.25)
    cg_q75 = cg[cg > 0].quantile(0.75)
    def cg_bin(val):
        if val == 0:
            return "0"
        elif val < cg_q25:
            return "small"
        elif val < cg_q75:
            return "medium"
        else:
            return "high"
    X["capital-gain"] = cg.apply(cg_bin)

    # Capital-loss bins using quartiles (non-zero values only)
    cl = X["capital-loss"].astype(float)
    cl_q25 = cl[cl > 0].quantile(0.25)
    cl_q75 = cl[cl > 0].quantile(0.75)
    def cl_bin(val):
        if val == 0:
            return "0"
        elif val < cl_q25:
            return "small"
        elif val < cl_q75:
            return "medium"
        else:
            return "high"
    X["capital-loss"] = cl.apply(cl_bin)

    # Fnlwgt bins using quartiles on log-transformed values
    X["fnlwgt"] = np.log1p(X["fnlwgt"].astype(float))
    fnlwgt_bins = X["fnlwgt"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    fnlwgt_labels = ["Q1", "Q2", "Q3", "Q4"]
    X["fnlwgt"] = pd.cut(X["fnlwgt"], bins=fnlwgt_bins, labels=fnlwgt_labels, include_lowest=True)

    return X

import pandas as pd

def preprocess_mushroom_dataset(df):
    # Rename column if needed
    if 'bruises%3F' in df.columns:
        df = df.rename(columns={'bruises%3F': 'bruises'})

    # Mapping for categorical variables
    value_maps = {
        'class': {'e': 'edible', 'p': 'poisonous'},
        'cap-shape': {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'},
        'cap-surface': {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'},
        'cap-color': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'bruises': {'t': 'yes', 'f': 'no'},
        'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'},
        'gill-attachment': {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'},
        'gill-spacing': {'c': 'close', 'w': 'crowded', 'd': 'distant'},
        'gill-size': {'b': 'broad', 'n': 'narrow'},
        'gill-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
        'stalk-root': {'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', '?': 'missing'},
        'stalk-surface-above-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
        'stalk-surface-below-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
        'stalk-color-above-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'stalk-color-below-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'veil-type': {'p': 'partial', 'u': 'universal'},
        'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
        'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
        'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'},
        'spore-print-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green', 'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
        'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
        'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'},
    }

    # Apply mappings
    for col, mapping in value_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df
def generate_aleph_header_lines():
    """
    Returns the first two lines of the Aleph preface header.
    """
    return "\n".join([
        ":- use_module(library(aleph)).",  # Load Aleph library
        ":- aleph."                        # Initialize Aleph
    ])

def generate_aleph_modes(pred_label):
    """
    Returns the other three lines (mode declarations and determination) of the Aleph preface header.
    """
    return "\n".join([
        f":- modeh(*, {pred_label}(+id)).",  # Head mode: target predicate
        ":- modeb(*, feature(+id, #feature, #value)).", # Body mode: features
        f":- determination({pred_label}/1, feature/3)."  # Allow features in rules for pred_label
    ])
