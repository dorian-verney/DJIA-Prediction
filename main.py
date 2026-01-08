import argparse
from argparse import ArgumentParser
from pathlib import Path


from src.experiments.run_experiments import run_experiments
config_base = Path(__file__).resolve().parent / "models"

def _available_models_by_data_type() -> dict[str, list[str]]:
    data_types = ["numerical", "textual"]
    result: dict[str, list[str]] = {}
    for dtype in data_types:
        dtype_dir = config_base / dtype
        if dtype_dir.exists():
            result[dtype] = sorted(p.stem for p in dtype_dir.glob("*.yaml"))
        else:
            result[dtype] = []
    return result   


AVAILABLE_MODELS = _available_models_by_data_type()
AVAILABLE_MODELS_HELP = "\n".join(
    f"  {dtype}: {', '.join(models) if models else 'none'}"
    for dtype, models in AVAILABLE_MODELS.items()
)

parser = ArgumentParser(
    description="Run an experiment with a registered model configuration.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=False,
    help=f"Model name. Available per data type:\n{AVAILABLE_MODELS_HELP}",
)
parser.add_argument(
    "-dtype",
    "--data_type",
    type=str,
    choices=["numerical", "textual"],
    required=True,
    help="Data type.",
)


args = parser.parse_args()

if __name__ == "__main__":
    available_for_dtype = AVAILABLE_MODELS.get(args.data_type, [])
    if args.model and args.model not in available_for_dtype:
        parser.error(
            f"Unknown model '{args.model}' for data type '{args.data_type}'. "
            f"Available: {', '.join(available_for_dtype)}"
        )
  
    config_path = config_base / args.data_type / f"{args.model}.yaml" 
    if args.data_type:
        config_path = config_base / args.data_type
        if args.model:
            config_path = config_path / f"{args.model}.yaml"


    if not config_path.exists():
        parser.error(f"Config file '{config_path}' not found.")

    print("---------- Running experiments ----------")

    run_experiments(config_path)
    print("\n---------- Experiments finished ----------\n")