import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help = "Location for Configuration File"
    )
    parser.add_argument(
        "-m", "--make", action = "store_true",
        help = "Create Default Configuration JSON"
    )
    parser.add_argument(
        "-l", "--log-file", help = "File Logger Name"
    )
    return parser.parse_args()

def defaults()

if __name__ == "__main__":
    parser = parse_args()
    if parser.make:
