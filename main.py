import numpy as np
import pandas as pd
import matplotlib as mplt
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import itg
import math

def loadData(file: str) -> np.NDArray:
    return pd.read_excel(file).to_numpy()

#define our cell system model for simulation
def cellSystem(t, x):
    #model parameters
    r_s = 1
    r_R = 1
    d_s = 1
    d_R = 1
    a_1 = 1
    delay_1 = 1
    delay_2 = 1

    #dS equation. (x1)
    dx1_dt = r_s * x1 * d_s * (1-math.e ** -(delay_1 * t)) * x1 - a_1 (1 - math.e ** - (delay_2*t)) * x1
    #dR equation (x2)
    dx2_dt = r_R * x2 + a_1(1 - math.e ** -(delay_2 * t)) * x1 - d_R(1-e^-(delay_1 * t)) * x2
    return [dx1_dt, dx2_dt]

def simulateSystem():
    solver = itg.RK45(cellSystem, 0, [0.25,0.25], 5000, max_step=0.05)
    soly = solver.y.reshape(1, 2)
    #run 2000 steps on the solver
    for i in range(2000):
        solver.step()
        solyn = solver.y.reshape(1, 2)
        soly = np.concatenate((soly, solyn))

def main():
    dt = loadData("pcbi.1007688.s007.xlsx")
    print(dt)

    #Pysindy Settings
    differentiation_method = ps.FiniteDifference(order=2)
    feature_library = ps.PolynomialLibrary(degree=3)

    #Cut off threshold for terms
    optimizer = ps.STLSQ(threshold=0.1)

    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=feature_library,
        optimizer=optimizer,
    )
    
    #Fit pysindy model to data
    model.fit(X, t=t, feature_names=["x", "y"])

    #Print fit model equations
    model.print()
