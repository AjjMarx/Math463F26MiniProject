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
    div = np.zeros((1000, 3))
    for i in range(1000):
        #print(sol.y.T[i])
        s = cellSystem(sol.t[i], sol.y.T[i], r_s, d_s, alpha, r_r, d_r, delay)
        div[i] = [s[0], s[1], 0.1]
    return (sol, div)

def plotFig(fig, axs, left, down, X, sol, name):
    axs[down][left].plot(X, sol[0], label='Model: S', color='green', linestyle='--')
    axs[down][left].plot(X, sol[1], label='Model: R', color='black', linestyle=':')
    axs[down][left].plot(X, sol[0] + sol[1], label='Model: S+R', color='red', linestyle='-')
    axs[down][left].set_title(name)
    if (down == 2):
        axs[down][left].set_xlabel("Time (hr)")
    if (left == 0):
        axs[down][left].set_ylabel("Normalized Cell Count")
    #axs[down][left].legend()

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
            math.exp(state[1])*state[1],
            math.exp(state[2])
            ])

def library(states) -> np.ndarray:
    length = states.shape[1]
    functionCount = libraryFunctions([0, 0, 0]).shape[0]
    theta = np.zeros((length, functionCount))
    for i in range(length):
        theta[i] = libraryFunctions(states.T[i])

    return theta

def generateOut(states, coef) -> np.ndarray:
    length = states.shape[1]
    varCount = coef.shape[0]
    theta = np.zeros((length, varCount))
    theta[0][0] = 1
    theta[0][1] = 0
    for i in range(1, length):
        for j in range(varCount):
            theta[i][j] = theta[i-1][j] + 0.1*np.dot(libraryFunctions(states.T[i]), coef[j])

    return theta

def main():
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(8,8))#, shareX=True)

    solution3_2  = simulateSystem(0.0880, 0.6924, 0.1048, 0.0070, 0.0137, 0.01)
    solution_32  = simulateSystem(0.0722, 0.4012, 0.1963, 0.0415, 0.0699, 0.01)
    solution__32 = simulateSystem(0.0532, 0.1855, 0.2520, 0.0532, 0.0751, 0.01)

    solutions = [solution3_2, solution_32, solution__32]
    alph = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

    for j in range(3):
        print(solutions[j][0].y)
        print(np.vstack((solutions[j][0].y, solutions[j][0].t)))
        theta = library(np.vstack((solutions[j][0].y, solutions[j][0].t)))
        print(theta)
        for i in range(4):
            L1Reg = Lasso(alpha = alph[i], fit_intercept=False, max_iter=10000) 
            L1Reg.fit(theta, solutions[j][1])
            dat = generateOut(np.vstack((solutions[j][0].y, solutions[j][0].t)), L1Reg.coef_)
            plotFig(fig, axs, i + 1, j, solutions[j][0].t, dat.T, "3.2 µM, Reg. " + str(alph[i]))


    plotFig(fig, axs, 0, 0, solution3_2[0].t, solution3_2[0].y, "Original 3.2 µM")
    plotFig(fig, axs, 0, 1, solution_32[0].t, solution_32[0].y, "Original 0.32 µM")
    plotFig(fig, axs, 0, 2, solution__32[0].t, solution__32[0].y, "Original 0.032 µM")
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.25)
    plt.show()


if __name__ == "__main__":
    main()
