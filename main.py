import numpy as np
import pandas as pd
import matplotlib as mplt
import pysindy as ps
import matplotlib.pyplot as plt
import scipy.integrate as itg
import math
from sklearn.linear_model import Lasso

#define our cell system model for simulation
def cellSystem(t, state, r_s, d_s, alpha, r_r, d_r, delay) -> tuple[float, float]:
    S = state[0]
    R = state[1]
    
    dS_dt = r_s*S - d_s*(1 - math.exp(-delay*t))*S - alpha*(1 - math.exp(-delay*t))*S
    dR_dt = r_r*R + alpha*(1 - math.exp(-delay*t))*S - d_r*(1 - math.exp(-delay*t))*R
    return [dS_dt, dR_dt]

def simulateSystem(r_s, d_s, alpha, r_r, d_r, delay) -> tuple(np.ndarray, np.ndarray):
    sol = itg.solve_ivp(fun = cellSystem, t_span = (0,100), y0 = [1, 0], t_eval = np.linspace(0, 100, 1000), args=(r_s, d_s, alpha, r_r, d_r, delay))
    div = np.zeros((1000, 2))
    for i in range(1000):
        #print(sol.y.T[i])
        div[i] = cellSystem(sol.t[i], sol.y.T[i], r_s, d_s, alpha, r_r, d_r, delay)

    return (sol, div)

def plotFig(fig, axs, down, sol, name):
    axs[down].plot(sol.t, sol.y[0], label='Model: S', color='green', linestyle='--')
    axs[down].plot(sol.t, sol.y[1], label='Model: R', color='black', linestyle=':')
    axs[down].plot(sol.t, sol.y[0] + sol.y[1], label='Model: S+R', color='red', linestyle='-')
    axs[down].set_title(name)
    axs[down].set_xlabel("Time (hr)")
    axs[down].set_ylabel("Normalized Cell Count")
    axs[down].legend()

def libraryFunctions(state) -> np.ndarray:
    return np.array([
            1,
            state[0],
            state[1],
            math.exp(state[0]),
            math.exp(state[1]),
            math.exp(state[0])*state[0],
            math.exp(state[1])*state[0],
            math.exp(state[0])*state[1],
            math.exp(state[1])*state[1]
            ])

def library(states) -> np.ndarray:
    length = states.shape[1]
    functionCount = libraryFunctions([0, 0]).shape[0]
    theta = np.zeros((length, functionCount))
    for i in range(length):
        theta[i] = libraryFunctions(states.T[i])

    return theta


def main():
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,8))#, shareX=True)

    solution3_2  = simulateSystem(0.0880, 0.6924, 0.1048, 0.0070, 0.0137, 0.01)
    solution_32  = simulateSystem(0.0722, 0.4012, 0.1963, 0.0415, 0.0699, 0.01)
    solution__32 = simulateSystem(0.0532, 0.1855, 0.2520, 0.0532, 0.0751, 0.01)

    theta = library(solution3_2[0].y)
    print(solution3_2[1])
    
    #L1Reg = Lasso(alpha = 0.01, fit_intercept=False, max_iter=10000) 
    #l1Reg.fit(Theta, div)

    plotFig(fig, axs, 0, solution3_2[0], "3.2 µM")
    axs[0].plot(solution3_2[0].t, 10*solution3_2[1].T[0], label='Model: S', color='green', linestyle='-')
    plotFig(fig, axs, 1, solution_32[0], "0.32 µM")
    plotFig(fig, axs, 2, solution__32[0], "0.032 µM")
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    main()
