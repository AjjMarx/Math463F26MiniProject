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
    sol = itg.solve_ivp(fun = cellSystem, t_span = (0,300), y0 = [1, 0], t_eval = np.linspace(0, 300, 3000), args=(r_s, d_s, alpha, r_r, d_r, delay))
    div = np.zeros((3000, 3))
    for i in range(3000):
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
            1,
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
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8,8))#, shareX=True)

    solutionT = simulateSystem(0.033, 0.158, 0.08, 0.0415, 0.0876, 0.01)
    axs[0][0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[0][1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[1][0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[1][1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[0][2].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[1][2].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    axs[0][2].set_yticks([0, 0.033, -0.158 -0.08])
    axs[1][2].set_yticks([0, 0.0493, 0.158, -0.0876, 0.08])

    axs[0][0].set_ylim(-0.1, 2) 
    axs[1][0].set_ylim(-20, 20)
    axs[0][1].set_ylim(-10, 10)
    axs[1][1].set_ylim(1e-7, 1e1)
    axs[1][0].set_yscale('symlog')
    axs[0][1].set_yscale('symlog')
    axs[1][1].set_yscale('log')

    axs[0][0].set_title("Challenge Explicit Integration")
    axs[0][0].set_xlabel("Time (hr)")
    axs[0][0].set_ylabel("Normalized Cell Count")
    axs[1][0].set_title("SINDy Fits")
    axs[1][0].set_ylabel("Normalized Cell Count")
    axs[1][0].set_xlabel("Time (hr)")
    axs[0][1].set_title("SINDy Fit Residuals")
    axs[0][1].set_xlabel("Time (hr)")
    axs[1][1].set_title("1-R²")
    axs[1][1].set_xlabel("Interval Right Bound (hr)")
    axs[1][1].set_ylabel("1-R^2 (log)")
    axs[0][2].set_title("Coefficients of S'(t)")
    axs[1][2].set_title("Coefficients of R'(t)")
    axs[0][2].set_xlabel("Interval Right Bound (hr)")
    axs[1][2].set_xlabel("Interval Right Bound (hr)")
    axs[0][2].set_ylabel("value")
    axs[1][2].set_ylabel("value")
    
    axs[0][0].plot(solutionT[0].t, solutionT[0].y[0], label='Model: S', color='green', linestyle='-')
    axs[0][0].plot(solutionT[0].t, solutionT[0].y[1], label='Model: R', color='black', linestyle='-')
    axs[0][0].plot(solutionT[0].t, solutionT[0].y[0] + solutionT[0].y[1], label='Model: S+R', color='red', linestyle='-')

    np.set_printoptions(suppress=True)
    cmapA = plt.colormaps['summer']
    cmapB = plt.colormaps['autumn']
    cmapC = plt.colormaps['winter']

    mean = [
            (solutionT[0].y[0]).sum() / 300,
            (solutionT[0].y[1]).sum() / 300,
            (solutionT[0].y[0] + solutionT[0].y[1]).sum() / 300
            ]
    print(mean)
    SS_tot = [
             (((solutionT[0].y[0]) - mean[0])**2).sum() /300,
             (((solutionT[0].y[1]) - mean[1])**2).sum() /300,
             (((solutionT[0].y[0] + solutionT[0].y[1]) - mean[2])**2).sum() /300
            ]
    #print(SS_tot)

    for i in range(49):
        intervalY = solutionT[0].y[:,0:(i+1)*60]
        intervalT = solutionT[0].t[0:(i+1)*60]
        theta = library(np.vstack((intervalY, intervalT)))
        L1Reg = Lasso(alpha = 0.00001, fit_intercept=False, max_iter=10000)
        L1Reg.fit(theta, solutionT[1][0:(i+1)*60])
        L1Reg.coef_[np.abs(L1Reg.coef_) < 0.00001] = 0.0
        dat = generateOut(np.vstack((solutionT[0].y, solutionT[0].t)), L1Reg.coef_)

        axs[1][0].plot(solutionT[0].t, dat.T[0], label='Model: S'+str(i), c=cmapA(1-i/20), linestyle=':')
        axs[1][0].plot(solutionT[0].t, dat.T[1], label='Model: R'+str(i), c=cmapC(1-i/20), linestyle=':')
        axs[1][0].plot(solutionT[0].t, dat.T[0]+dat.T[1], label='Model: (S+R)'+str(i), c=cmapB(1-i/20), linestyle=':')
        axs[0][1].plot(solutionT[0].t, (solutionT[0].y[0]) - (dat.T[0]), label='Model: S'+str(i), c=cmapA(1-i/50), linestyle='-')
        axs[0][1].plot(solutionT[0].t, (solutionT[0].y[1]) - (dat.T[1]), label='Model: R'+str(i), c=cmapC(1-i/50), linestyle='-')
        axs[0][1].plot(solutionT[0].t, (solutionT[0].y[0] + solutionT[0].y[1]) - (dat.T[0]+dat.T[1]), label='Model: (S+R)'+str(i), c=cmapB(1-i/50), linestyle='-')
        SS_res = [
                (((solutionT[0].y[0]) - (dat.T[0]))**2).sum() / 300,
                (((solutionT[0].y[1]) - (dat.T[1]))**2).sum() / 300,
                (((solutionT[0].y[0] + solutionT[0].y[1]) - (dat.T[0]+dat.T[1]))**2).sum() / 300
                ]
        #print("SS_res: ", SS_res, ", SS_tot: ", SS_tot, ", R^2: ", 1 -  SS_res/ SS_tot)
        X = np.full((L1Reg.coef_[0].shape[0]), (i+1)*6)
        axs[1][1].scatter((i+1)*6, SS_res[0]/ SS_tot[0], color=cmapA(1-i/50))
        axs[1][1].scatter((i+1)*6, SS_res[1]/ SS_tot[1], color=cmapC(1-i/50))
        axs[1][1].scatter((i+1)*6, SS_res[2]/ SS_tot[2], color=cmapB(1-i/50))
        axs[0][2].scatter(X, L1Reg.coef_[0], color=cmapA(1-i/50))
        axs[1][2].scatter(X, L1Reg.coef_[1], color=cmapC(1-i/50))

    fig.subplots_adjust(wspace=0.3)
    plt.show()

if __name__ == "__main__":
    main()
