from datetime import datetime
import sys

class Logger:
    def __init__(self, filename, opening_mode = "w"):
        self.filename = filename
        self.outfile = open(filename, opening_mode)
        self.initialized = True
        self.streams = [sys.stdout, self.outfile]

    def write(self, message, severity=""):
        severity = f"[{severity}] " if severity != "" else severity
        for stream in self.streams:
            stream.write(f"[{str(datetime.now())}]: {severity.upper()}{message}\n")
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def close(self):
        self.streams[-1].close()