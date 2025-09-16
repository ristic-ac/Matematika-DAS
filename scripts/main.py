import os
import sys
from pyswip import Prolog
from rule_check import compute_confusion_for_folder, plot_confusion_matrix_distillate, print_confusion_matrix_distillate
from train_and_export import train_and_export_aleph_single
from rules import clean_aleph_program

def build_aleph_modes(num_samples):
    """
    Build ALEPH_MODES dynamically, scaling noise as noise_base * factor.
    """

    return [
        {
            
            "folder": "fast_bounded",
            "settings": (
                ":- aleph_set(search, ibs).\n"
                ":- aleph_set(openlist, 15).\n"
                ":- aleph_set(nodes, 1500).\n"
                ":- aleph_set(evalfn, laplace).\n"
                ":- aleph_set(minacc, 0.55).\n"
                f":- aleph_set(noise, {int(num_samples * 0.02)}).\n"
            )
        },
        {
            "folder": "fast_precise",
            "settings": (
                ":- aleph_set(search, heuristic).\n"
                ":- aleph_set(evalfn, accuracy).\n"
                ":- aleph_set(nodes, 3000).\n"
                ":- aleph_set(clauselength, 4).\n"
                ":- aleph_set(minacc, 0.7).\n"
                ":- aleph_set(minpos, 3).\n"
                f":- aleph_set(noise, {int(num_samples * 0.02)}).\n"
            )
        },
        {
            "folder": "iterative_beam_search",
            "settings": (
                ":- aleph_set(search, ibs).\n"
                ":- aleph_set(openlist, 10).\n"
                ":- aleph_set(evalfn, compression).\n"
                ":- aleph_set(nodes, 4000).\n"
                ":- aleph_set(clauselength, 4).\n"
                ":- aleph_set(minpos, 5).\n"
                f":- aleph_set(noise, {int(num_samples * 0.05)}).\n"
            )
        }
        ,
        {
            "folder": "breadth_first_laplace",
            "settings": (
            ":- aleph_set(search, bf).\n"
            ":- aleph_set(evalfn, laplace).\n"
            ":- aleph_set(nodes, 5000).\n"
            ":- aleph_set(clauselength, 3).\n"
            ":- aleph_set(minacc, 0.80).\n"
            ":- aleph_set(minpos, 3).\n"
            f":- aleph_set(noise, {int(num_samples * 0.05)}).\n"
            )
        }
    ]


def run_aleph_with_files(model_type, dataset, aleph_settings):
    """
    Run Aleph by consulting only {dataset}.pl in the corresponding model_type folder, as in swi.sh.
    """

    aleph_folder, _ = list(aleph_settings)

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
        metrics = compute_confusion_for_folder(base_dir, dataset)
        print_confusion_matrix_distillate(metrics, base_dir)
        plot_confusion_matrix_distillate(metrics, outdir=base_dir, model_type=model, dataset=dataset)
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
        mode_index = int(sys.argv[4]) if len(sys.argv) > 4 else None  # <--- NEW ARG

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

    testing_size = None
    if dataset == "mushroom":
        testing_size = 6657
    elif dataset == "adult":
        testing_size = 0

    for model in models:
        aleph_modes = build_aleph_modes(testing_size)

        if action == "train":
            if mode_index is not None:
                mode = aleph_modes[mode_index]
                train_and_export_aleph_single(model, dataset, (mode["folder"], mode["settings"]))
            else:
                for mode in aleph_modes:
                    train_and_export_aleph_single(model, dataset, (mode["folder"], mode["settings"]))

        elif action == "aleph":
            if mode_index is not None:
                mode = aleph_modes[mode_index]
                run_aleph_with_files(model, dataset, (mode["folder"], mode["settings"]))
            else:
                for mode in aleph_modes:
                    run_aleph_with_files(model, dataset, (mode["folder"], mode["settings"]))

        elif action == "both":
            if mode_index is not None:
                mode = aleph_modes[mode_index]
                train_and_export_aleph_single(model, dataset, (mode["folder"], mode["settings"]))
                run_aleph_with_files(model, dataset, (mode["folder"], mode["settings"]))
            else:
                for mode in aleph_modes:
                    train_and_export_aleph_single(model, dataset, (mode["folder"], mode["settings"]))
                    run_aleph_with_files(model, dataset, (mode["folder"], mode["settings"]))
        else:
            print("Invalid action. Use one of: train, aleph, both")
            sys.exit(1)
