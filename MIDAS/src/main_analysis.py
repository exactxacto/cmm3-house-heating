# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:06:28 2025

@author: yaya0
"""

# Start
from pathlib import Path
import pandas as pd


# Find the repository root
def repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    else:
        return Path.cwd()
    
    
# Loads Excel data from the 'data' folder and specified sheet 'Simple'.
def load_data(filename="EdiRadTempJan.xlsx", sheet_name="Simple"):
    base = repo_root()
    data_path = base / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    # Read only the needed sheet
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    return df

#Define Run Read first 10 lines
def main():
    df = load_data("EdiRadTempJan.xlsx", sheet_name="Simple")
    print("Loaded data sample (first 10 rows):\n")
    print(df.head(20))
    
#run
if __name__ == "__main__":
    main()