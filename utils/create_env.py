from pathlib import Path

if __name__ == "__main__":
    with open(".env", "w") as f:
        f.writelines(f"PYTHONPATH={str(Path.cwd())}")
