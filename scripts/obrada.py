import os
import json
import pandas as pd

BASE_DIR = "../outputs"
OUTPUT_FILE = "all_results.json"


def load_feature_importances(path):
    try:
        df = pd.read_csv(path)
        return df.set_index(df.columns[0])[df.columns[1]].to_dict()
    except Exception as e:
        print(f"Error loading feature importances {path}: {e}")
        return {}


def load_feature_columns(path):
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading feature columns {path}: {e}")
        return []


def load_classification_report(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading classification report {path}: {e}")
        return {}


def load_best_params(path):
    try:
        params = {}
        with open(path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    params[key.strip()] = value.strip()
        return params
    except Exception as e:
        print(f"Error loading best params {path}: {e}")
        return {}


def main():
    results = {}

    for dataset in os.listdir(BASE_DIR):
        dataset_path = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(dataset_path):
            continue

        results[dataset] = {}

        for model in ["dt", "rf", "xgb"]:
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue

            best_params = load_best_params(
                os.path.join(model_path, "best_params.txt"))
            feature_importances = load_feature_importances(
                os.path.join(model_path, "feature_importances.csv"))
            feature_columns = load_feature_columns(
                os.path.join(model_path, "feature_columns.txt"))
            classification_report = load_classification_report(
                os.path.join(model_path, "classification_report.json"))

            results[dataset][model] = {
                "best_params": best_params,
                "feature_importances": feature_importances,
                "feature_columns": feature_columns,
                "classification_report": classification_report,
            }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved consolidated results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
