from pathlib import Path
import pandas as pd

def iterdirs(path, glossary, depth = 2):
    if depth == 0:
        return glossary
        
    for i in path.glob("*"):
        if i.is_dir():
            foldername = str(i.relative_to(Path.cwd()))
            if foldername not in glossary.Path.tolist():
                glossary = glossary._append(
                    {"Path": foldername, "Description": ""}, 
                    ignore_index = True
                )
            glossary = iterdirs(path / i, glossary, depth = depth - 1)

    return glossary
    
glossary = pd.read_excel(Path.cwd() / "data/Glossary.xlsx")
glossary = iterdirs(Path.cwd() / "data", glossary, depth = 2)

with pd.ExcelWriter(Path.cwd() / "data/Glossary.xlsx") as f:
    glossary.to_excel(f, index = False)