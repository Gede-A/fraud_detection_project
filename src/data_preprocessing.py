#Define a function for data loading and structure to data frame
import pandas as pd
import os
def load_data(filepath:str)->pd.DataFrame:
    if not os.path.exists(filepath):
         raise FileNotFoundError(f"File not found at: {filepath}")
    else:
        return pd.read_csv(filepath)

    