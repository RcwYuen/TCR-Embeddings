import subprocess
import argparse
import os


LS_VALID_REDUCTION = ["no-reduction", "johnson-lindenstarauss", "autoencoder"]

LS_VALID_ENCODING = [
    "atchley",
    "kidera",
    "rand",
    "aaprop",
    "sceptr-default",
    "sceptr-tiny",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reduction",
        help="Reduction Method, Valid Inputs are: " + "; ".join(LS_VALID_REDUCTION),
    )
    parser.add_argument(
        "-e",
        "--encoding",
        help="Encoding Method, Valid Inputs are: " + "; ".join(LS_VALID_ENCODING),
    )
    return parser.parse_args()


if __name__ == "__main__":
    kfolds = 5
    working_dir = "path/to/your/working/directory"
    current_shell = os.environ.get("SHELL", "/bin/bash")

    parser = parse_args()
    encoding = parser.encoding.lower()
    reduction = parser.reduction.lower()

    assert encoding in LS_VALID_ENCODING, "Please input a valid encoding method."
    assert reduction in LS_VALID_REDUCTION, "Please input a valid reduction method."

    for i in range(kfolds):
        fname = f"tcr_embeddings/configs/{encoding}/{reduction}/kfold-{i}.json"  # change to your config names
        subprocess.run(
            f"screen -dmS {encoding}-{reduction}_{i}",
            shell=True,
            executable=current_shell,
        )
        subprocess.run(
            f'screen -S {encoding}-{reduction}_{i} -X stuff "python -m tcr_embeddings.trainer --config {fname}\\n"',
            shell=True,
            executable=current_shell,
        )
