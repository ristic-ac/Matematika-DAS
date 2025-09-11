import os
import sys
from pyswip import Prolog
from train_and_export import train_and_export_aleph_single
from rules import clean_aleph_program

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
