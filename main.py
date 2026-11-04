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

def libraryFunctionsA(state) -> np.ndarray:
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

libraryStringsA = ["(1)", 
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

def libraryFunctionsB(state) -> np.ndarray:
    return np.array([
            1,
            state[0],
            state[1],
            math.cos(state[0]),
            math.sin(state[0]),
            math.cos(state[1]),
            math.sin(state[1]),
            math.cos(state[0])*state[0],
            math.sin(state[0])*state[0],
            math.cos(state[0])*state[1],
            math.sin(state[0])*state[1],
            math.cos(state[0])*state[0]*state[1],
            math.sin(state[0])*state[0]*state[1],
            math.cos(state[1])*state[0],
            math.sin(state[1])*state[0],
            math.cos(state[1])*state[1],
            math.sin(state[1])*state[1],
            math.cos(state[1])*state[0]*state[1],
            math.sin(state[1])*state[0]*state[1],
            math.cos(state[0]/2),
            math.sin(state[0]/2),
            math.cos(state[1]/2),
            math.sin(state[1]/2),
            math.cos(state[0]/2)*state[0],
            math.sin(state[0]/2)*state[0],
            math.cos(state[0]/2)*state[1],
            math.sin(state[0]/2)*state[1],
            math.cos(state[0]/2)*state[0]*state[1],
            math.sin(state[0]/2)*state[0]*state[1],
            math.cos(state[1]/2)*state[0],
            math.sin(state[1]/2)*state[0],
            math.cos(state[1]/2)*state[1],
            math.sin(state[1]/2)*state[1],
            math.cos(state[1]/2)*state[0]*state[1],
            math.sin(state[1]/2)*state[0]*state[1],
            math.cos(state[0]/10),
            math.sin(state[0]/10),
            math.cos(state[1]/10),
            math.sin(state[1]/10),
            math.cos(state[0]/10)*state[0],
            math.sin(state[0]/10)*state[0],
            math.cos(state[0]/10)*state[1],
            math.sin(state[0]/10)*state[1],
            math.cos(state[0]/10)*state[0]*state[1],
            math.sin(state[0]/10)*state[0]*state[1],
            math.cos(state[1]/10)*state[0],
            math.sin(state[1]/10)*state[0],
            math.cos(state[1]/10)*state[1],
            math.sin(state[1]/10)*state[1],
            math.cos(state[1]/10)*state[0]*state[1],
            math.sin(state[1]/10)*state[0]*state[1],
            math.cos(state[0]/15),
            math.sin(state[0]/15),
            math.cos(state[1]/15),
            math.sin(state[1]/15),
            math.cos(state[0]/15)*state[0],
            math.sin(state[0]/15)*state[0],
            math.cos(state[0]/15)*state[1],
            math.sin(state[0]/15)*state[1],
            math.cos(state[0]/15)*state[0]*state[1],
            math.sin(state[0]/15)*state[0]*state[1],
            math.cos(state[1]/15)*state[0],
            math.sin(state[1]/15)*state[0],
            math.cos(state[1]/15)*state[1],
            math.sin(state[1]/15)*state[1],
            math.cos(state[1]/15)*state[0]*state[1],
            math.sin(state[1]/15)*state[0]*state[1],
            math.cos(state[0]/30),
            math.sin(state[0]/30),
            math.cos(state[1]/30),
            math.sin(state[1]/30),
            math.cos(state[0]/30)*state[0],
            math.sin(state[0]/30)*state[0],
            math.cos(state[0]/30)*state[1],
            math.sin(state[0]/30)*state[1],
            math.cos(state[0]/30)*state[0]*state[1],
            math.sin(state[0]/30)*state[0]*state[1],
            math.cos(state[1]/30)*state[0],
            math.sin(state[1]/30)*state[0],
            math.cos(state[1]/30)*state[1],
            math.sin(state[1]/30)*state[1],
            math.cos(state[1]/30)*state[0]*state[1],
            math.sin(state[1]/30)*state[0]*state[1],
            ])

libraryStringsB = ["(1)",
                  "(S)", 
                  "(R)", 
                  "(cos(S))",
                  "(sin(S))",
                  "(cos(R))",
                  "(sin(R))",
                  "(cos(S))S",
                  "(sin(S))S",
                  "(cos(S))R",
                  "(sin(S))R",
                  "(cos(S))SR",
                  "(sin(S))SR",
                  "(cos(R))S",
                  "(sin(R))S",
                  "(cos(R))R",
                  "(sin(R))R",
                  "(cos(R))SR",
                  "(sin(R))SR",

                  "(cos(S/2))",
                  "(sin(S/2))",
                  "(cos(R/2))",
                  "(sin(R/2))",
                  "(cos(S/2))S",
                  "(sin(S/2))S",
                  "(cos(S/2))R",
                  "(sin(S/2))R",
                  "(cos(S/2))SR",
                  "(sin(S/2))SR",
                  "(cos(R/2))S",
                  "(sin(R/2))S",
                  "(cos(R/2))R",
                  "(sin(R/2))R",
                  "(cos(R/2))SR",
                  "(sin(R/2))SR",

                  "(cos(S/10))",
                  "(sin(S/10))",
                  "(cos(R/10))",
                  "(sin(R/10))",
                  "(cos(S/10))S",
                  "(sin(S/10))S",
                  "(cos(S/10))R",
                  "(sin(S/10))R",
                  "(cos(S/10))SR",
                  "(sin(S/10))SR",
                  "(cos(R/10))S",
                  "(sin(R/10))S",
                  "(cos(R/10))R",
                  "(sin(R/10))R",
                  "(cos(R/10))SR",
                  "(sin(R/10))SR",

                  "(cos(S/15))",
                  "(sin(S/15))",
                  "(cos(R/15))",
                  "(sin(R/15))",
                  "(cos(S/15))S",
                  "(sin(S/15))S",
                  "(cos(S/15))R",
                  "(sin(S/15))R",
                  "(cos(S/15))SR",
                  "(sin(S/15))SR",
                  "(cos(R/15))S",
                  "(sin(R/15))S",
                  "(cos(R/15))R",
                  "(sin(R/15))R",
                  "(cos(R/15))SR",
                  "(sin(R/15))SR",

                  "(cos(S/30))",
                  "(sin(S/30))",
                  "(cos(R/30))",
                  "(sin(R/30))",
                  "(cos(S/30))S",
                  "(sin(S/30))S",
                  "(cos(S/30))R",
                  "(sin(S/30))R",
                  "(cos(S/30))SR",
                  "(sin(S/30))SR",
                  "(cos(R/30))S",
                  "(sin(R/30))S",
                  "(cos(R/30))R",
                  "(sin(R/30))R",
                  "(cos(R/30))SR",
                  "(sin(R/30))SR",
                  ]

def libraryFunctionsC(state) -> np.ndarray:
    return np.array([
            state[0],
            state[1],
            (1-math.exp(-0.01*state[2]))*state[0],
            (1-math.exp(-0.01*state[2]))*state[1]
            ])

def library(states) -> np.ndarray:
    length = states.shape[1]
    functionCount = libraryFunctionsC([0, 0, 0]).shape[0]
    theta = np.zeros((length, functionCount))
    for i in range(length):
        theta[i] = libraryFunctionsC(states.T[i])

    return theta

def generateOut(states, coef) -> np.ndarray:
    length = states.shape[1]
    varCount = coef.shape[0]
    theta = np.zeros((length, varCount))
    theta[0][0] = 1
    theta[0][1] = 0
    for i in range(1, length):
        for j in range(varCount):
            theta[i][j] = theta[i-1][j] + 0.1*np.dot(libraryFunctionsC(states.T[i]), coef[j])

    return theta

def main():
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8,8))#, shareX=True)
    df = pd.read_excel("pcbi.1007688.s007.xlsx")
    dataplot = df.to_numpy().T;

    solution3_2  = simulateSystem(0.0880, 0.6924, 0.1048, 0.0070, 0.0137, 0.01)
    solutionI1   = simulateSystem(0.0800, 0.5452, 0.1511, 0.0244, 0.0421, 0.01)
    solution_32  = simulateSystem(0.0722, 0.4012, 0.1963, 0.0415, 0.0699, 0.01)
    solutionI2   = simulateSystem(0.0626, 0.2922, 0.2244, 0.0474, 0.07253, 0.01)
    solution__32 = simulateSystem(0.0532, 0.1855, 0.2520, 0.0532, 0.0751, 0.01)
    #solutionT = simulateSystem(0.033, 0.158, 0.08, 0.043, 0.0902, 0.01)

    solutions = [solution3_2, solutionI1, solution_32, solutionI2, solution__32]
    alph = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    titles = ["3.2", "1.0", "0.32", "0.1", "0.032"]

    np.set_printoptions(suppress=True)
    for j in range(5):
        theta = library(np.vstack((solutions[j][0].y, solutions[j][0].t)))
        #for i in range(4):
        L1Reg = Lasso(alpha = 0.00001, fit_intercept=False, max_iter=10000)
        L1Reg.fit(theta, solutions[j][1])
        L1Reg.coef_[np.abs(L1Reg.coef_) < 0.001] = 0.0
        dat = generateOut(np.vstack((solutions[j][0].y, solutions[j][0].t)), L1Reg.coef_)
        scatter = dataplot[1 + 4*(5-j)][1:752] / 4
        for i in range(3):
            scatter = scatter + dataplot[2 + i + 4*(5-j)][1:752] / 4
        scatter = scatter/scatter[0]
        erbars = ((scatter - dataplot[1 + 4*(5-j)][1:752]/dataplot[1 + 4*(5-j)][1])**2)/4
        for i in range(3):
            erbars = erbars + ((scatter - dataplot[2 + i + 4*(5-j)][1:752]/dataplot[2 + i + 4*(5-j)][1])**2) / 4
        axs[j].errorbar(dataplot[0][1:752], scatter, yerr=np.sqrt(np.asarray(erbars, dtype=float)), label='Dataset', color='blue', elinewidth=0.1, linestyle='-')
        axs[j].plot(solutions[j][0].t, dat.T[0], label='Model: S', color='green', linestyle='--')
        axs[j].plot(solutions[j][0].t, dat.T[1], label='Model: R', color='black', linestyle=':')
        axs[j].plot(solutions[j][0].t, dat.T[0]+dat.T[1], label='Model: S+R', color='red', linestyle='-', zorder=3)
        axs[j].set_title(titles[j] + "  µM")
        print(L1Reg.coef_)

    #plt.subplots_adjust(hspace=0.75)
    plt.subplots_adjust(hspace=5)
    #fig.supylabel("Normalized Cell Count")
    #fig.supxlabel("Time (hr)")
    #fig.suptitle("SINDy Results with Original Dataset")
    #axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols = 5)
    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
