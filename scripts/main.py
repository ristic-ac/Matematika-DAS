import os
import sys
from pyswip import Prolog
from rule_check import compute_fidelity_confusion_for_folder, evaluate_folder, plot_confusion_matrix_distillate, print_confusion_matrix_distillate
from train_and_export import build_aleph_modes, train_and_export_aleph_single
from rules import clean_aleph_program

def run_aleph_with_files(model_type, dataset, aleph_folder):
    """
    Run Aleph by consulting only {dataset}.pl in the corresponding model_type folder, as in swi.sh.
    """

    # Build paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "outputs", dataset, model_type, aleph_folder)
    script_file = os.path.join(base_dir, f"{dataset}.pl")

    prolog = Prolog()

    # Load Aleph
    list(prolog.query("use_module(library(aleph))"))

    print(f"[Aleph] Consulting: {script_file}")
    list(prolog.query(f"consult('{script_file}')"))

    # Set testing data for .f and .n
    test_pos_file = os.path.join(base_dir, f"{dataset}_test.f")
    list(prolog.query(f"aleph_set(test_pos, '{test_pos_file}')"))
    test_neg_file = os.path.join(base_dir, f"{dataset}_test.n")
    list(prolog.query(f"aleph_set(test_neg, '{test_neg_file}')"))

    res = list(prolog.query("induce(Program)"))
    if res:
        program = res[0]['Program']
        cleaned = clean_aleph_program(program, out_path=os.path.join(base_dir, f"{dataset}_hypothesis.pl"))
        for c in cleaned:
            print(c)
        evaluate_folder(base_dir, dataset, model_type=model_type)
    else:
        print("No hypothesis term.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("No arguments provided, defaulting to: both dt mushroom")
        action = "both"
        model_arg = "dt"
        dataset = "mushroom"
        mode_index = None
    else:
        model_arg = sys.argv[1].lower()
        action = sys.argv[2].lower()
        dataset = sys.argv[3].lower()
        mode_index = sys.argv[4] if len(sys.argv) > 4 else None

    if model_arg == "all":
        models = ["dt", "rf", "xgb"]
    elif model_arg in ["dt", "rf", "xgb"]:
        models = [model_arg]
    else:
        print("Invalid model argument. Use one of: dt, rf, xgb, all")
        sys.exit(1)

    if dataset not in ["mushroom", "adult"]:
        print("Invalid dataset argument. Use one of: mushroom, adult")
        sys.exit(1)


    for model in models:
        aleph_modes = build_aleph_modes(dataset)
        if mode_index is not None:
            if mode_index not in aleph_modes:
                print(f"Invalid mode index: {mode_index}. Available modes: {list(aleph_modes.keys())}")
                sys.exit(1)
            modes_to_process = {mode_index: aleph_modes[mode_index]}
        else:
            modes_to_process = aleph_modes

        if action == "train":
            for mode_index, mode in modes_to_process.items():
                train_and_export_aleph_single(model, dataset, mode_index)

        elif action == "aleph":
            for mode_index, mode in modes_to_process.items():
                run_aleph_with_files(model, dataset, mode_index)

        elif action == "both":
            for mode_index, mode in modes_to_process.items():
                print(f"=== Model: {model}, Dataset: {dataset}, Mode: {mode_index} ===")
                train_and_export_aleph_single(model, dataset, mode_index)
                run_aleph_with_files(model, dataset, mode_index)

        else:
            print("Invalid action. Use one of: train, aleph, both")
            sys.exit(1)
