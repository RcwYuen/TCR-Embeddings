import sys
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, filename: str | Path, opening_mode: str = "w") -> None:
        self.filename: str | Path = filename
        self.outfile = open(filename, opening_mode)
        self.initialized: bool = True
        self.streams: list = [sys.stdout, self.outfile]

    def write(self, message: str, severity: str = "") -> None:
        severity = f"[{severity}] " if severity != "" else severity
        for stream in self.streams:
            stream.write(f"[{str(datetime.now())}]: {severity.upper()}{message}\n")
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def close(self) -> None:
        self.streams[-1].close()
