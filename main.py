import numpy as np
import pandas as pd
import matplotlib as mplt
import pysindy as ps
import matplotlib.pyplot as plt
import scipy.integrate as itg
import math
from sklearn.linear_model import Lasso
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    if (down == 4):
        axs[down][left].set_xlabel("Time (hr)")
    #if (left == 0):
        #axs[down][left].set_ylabel("Normalized Cell Count")

def libraryFunctions(state) -> np.ndarray:
    return np.array([
            1,
            state[0],
            state[1],
            state[2],
            state[0]*state[1],
            state[0]*state[2],
            state[1]*state[2],
            math.exp(state[0]),
            math.exp(state[1]),
            math.exp(state[0])*state[0],
            math.exp(state[1])*state[0],
            math.exp(state[0])*state[1],
            math.exp(state[1])*state[1],
            math.exp(state[2]),
            math.exp(state[2])*state[0],
            math.exp(state[2])*state[1],
            math.exp(state[2])*state[2],
            math.cos(state[2]),
            math.cos(state[2]/10.0),
            math.cos(state[2]/100.0),
            ])

libraryStrings = ["(1)", 
                  "(S)", 
                  "(R)", 
                  "(t)", 
                  "(SR)",
                  "(St)",
                  "(Rt)",
                  "(e^S)",
                  "(e^R)",
                  "(Se^S)",
                  "(Se^R)",
                  "(Re^S)",
                  "(Re^R)",
                  "(e^t)",
                  "(Se^t)",
                  "(Re^t)",
                  "(te^t)",
                  "(cos(t))",
                  "(cos(t/10))",
                  "(cos(t/100))"
                  ]

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
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(8,8))#, shareX=True)

    solution3_2  = simulateSystem(0.0880, 0.6924, 0.1048, 0.0070, 0.0137, 0.01)
    solutionI1   = simulateSystem(0.0800, 0.5452, 0.2244, 0.0474, 0.0725, 0.01)
    solution_32  = simulateSystem(0.0722, 0.4012, 0.1963, 0.0415, 0.0699, 0.01)
    solutionI2   = simulateSystem(0.0626, 0.2922, 0.1511, 0.0244, 0.0421, 0.01)
    solution__32 = simulateSystem(0.0532, 0.1855, 0.2520, 0.0532, 0.0751, 0.01)

    solutions = [solution3_2, solutionI1, solution_32, solutionI2, solution__32]
    alph = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    titles = ["3.2", "Inter. #1", "0.32", "Inter. #2", "0.032"]

    np.set_printoptions(suppress=True)
    for j in range(5):
        theta = library(np.vstack((solutions[j][0].y, solutions[j][0].t)))
        for i in range(4):
            L1Reg = Lasso(alpha = alph[i], fit_intercept=False, max_iter=10000)
            L1Reg.fit(theta, solutions[j][1])
            L1Reg.coef_[np.abs(L1Reg.coef_) < 0.00001] = 0.0
            dat = generateOut(np.vstack((solutions[j][0].y, solutions[j][0].t)), L1Reg.coef_)
            plotFig(fig, axs, i + 1, j, solutions[j][0].t, dat.T, titles[j] + " µM, Reg. " + str(alph[i]))
            print(titles[j] + " µM, Reg. " + str(alph[i]) + " coefficients:")
            #print(L1Reg.coef_)
            print("S' = ")
            for k in range(len(libraryStrings)):
                if math.fabs(L1Reg.coef_[0][k]) >= 0.0001:
                    print(f"{L1Reg.coef_[0][k]:.5f}" + libraryStrings[k]  + " + ", end="")
            print("")
            print("R' = ")
            for k in range(len(libraryStrings)):
                if math.fabs(L1Reg.coef_[1][k]) >= 0.0001:
                    print(f"{L1Reg.coef_[1][k]:.5f}"+ libraryStrings[k]  + " + ", end="")
            print("")


    plotFig(fig, axs, 0, 0, solution3_2[0].t, solution3_2[0].y, "Original 3.2 µM")
    plotFig(fig, axs, 0, 1, solutionI1[0].t, solutionI1[0].y, "Original Inter. #1")
    plotFig(fig, axs, 0, 2, solution_32[0].t, solution_32[0].y, "Original 0.32 µM")
    plotFig(fig, axs, 0, 3, solutionI2[0].t, solutionI2[0].y, "Original Inter. #2")
    plotFig(fig, axs, 0, 4, solution__32[0].t, solution__32[0].y, "Original 0.032 µM")
    plt.subplots_adjust(hspace=0.75)
    plt.subplots_adjust(wspace=0.25)
    fig.supylabel("Normalized Cell Count")
    fig.suptitle("Original Systems with L1-Regularized SINDy Results")
    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
