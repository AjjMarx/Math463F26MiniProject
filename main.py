import numpy as np
import pandas as pd
import matplotlib as mplt

def loadData(file: str) -> np.NDArray:
    return pd.read_excel(file).to_numpy()

def main():
    dt = loadData("pcbi.1007688.s007.xlsx")
    print(dt)
