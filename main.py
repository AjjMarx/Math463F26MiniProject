import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp


# Model parameters assigned by drug concentration (μM): 0.032, 0.32, 3.2

PARAMETERS = {
    0.032: {
        "R_S": 0.0532,
        "R_R": 0.0532,
        "D_S": 0.1855,
        "D_R": 0.0751,
        "ALPHA": 0.2520,
        "GAMMA_1": 0.01,
        "GAMMA_2": 0.01,
    },
    0.32: {
        "R_S": 0.0722,
        "R_R": 0.0415,
        "D_S": 0.4012,
        "D_R": 0.0699,
        "ALPHA": 0.1963,
        "GAMMA_1": 0.01,
        "GAMMA_2": 0.01,
    },
    3.2: {
        "R_S": 0.0880,
        "R_R": 0.0070,
        "D_S": 0.6924,
        "D_R": 0.0137,
        "ALPHA": 0.1048,
        "GAMMA_1": 0.01,
        "GAMMA_2": 0.01,
    },
}

# Select drug concentration
DRUG_CONCENTRATION = 0.32

# Assign corresponding model parameters
params = PARAMETERS[DRUG_CONCENTRATION]

#Sensitive Growth
R_S = params["R_S"]
#Resistant Growth
R_R = params["R_R"]
#Rate of sensitive death for drug at full effect
D_S = params["D_S"]
#Rate of resistant death for drug at full effect
D_R = params["D_R"]
#Rate of change from sensitive to resistant
ALPHA = params["ALPHA"]
#Speed of drug killing effect
GAMMA_1 = params["GAMMA_1"]
#Speed of drug resistant inducing effect
GAMMA_2 = params["GAMMA_2"]

# Editable simulation settings. Using original paper's time
INITIAL_STATE = np.array([1, 0], dtype=float)
START_TIME = 0.0
END_TIME = 100.0
TIME_STEP = 0.01

def cell_system(t: float, state: np.ndarray) -> np.ndarray:
    """Return dS/dt and dR/dt for the cell model."""
    susceptible, resistant = state
    transfer_rate = ALPHA * (1.0 - np.exp(-GAMMA_2 * t))
    susceptible_death = D_S * (1.0 - np.exp(-GAMMA_1 * t))
    resistant_death = D_R * (1.0 - np.exp(-GAMMA_1 * t))

    d_susceptible = R_S * susceptible - susceptible_death * susceptible - transfer_rate * susceptible
    d_resistant = R_R * resistant + transfer_rate * susceptible - resistant_death * resistant
    return np.array([d_susceptible, d_resistant])

def simulate_system() -> tuple[np.ndarray, np.ndarray]:
    """Simulate the model on a uniform time grid for SINDy."""
    time = np.arange(START_TIME, END_TIME + TIME_STEP / 2.0, TIME_STEP)
    solution = solve_ivp(
        cell_system,
        (START_TIME, END_TIME),
        INITIAL_STATE,
        t_eval=time,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution.t, solution.y.T

def fit_sindy_model(time: np.ndarray, data: np.ndarray) -> ps.SINDy:
    """Identify cell dynamics using a custom library of candidate functions."""

    # Candidate delay rates (h^-1)
    rates = [0.001, 0.01, 0.1]

    # Each feature receives S, R, and t, in that order.
    library_functions = [
        lambda s, r, t: s,
        lambda s, r, t: r,
    ]

    function_names = [
        lambda s, r, t: s,
        lambda s, r, t: r,
    ]

    # Create candidate exponential terms for both populations.
    for gamma in rates:

        def make_feature(g, population):
            if population == "S":
                return lambda s, r, t: s * (1 - np.exp(-g * t))
            else:
                return lambda s, r, t: r * (1 - np.exp(-g * t))

        def make_name(g, population):
            return lambda s, r, t: (
                f"{population}*(1-exp(-{g}*t))"
            )

        for population in ["S", "R"]:
            library_functions.append(
                make_feature(gamma, population)
            )
            function_names.append(
                make_name(gamma, population)
            )

    # Custom candidate function library
    library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=function_names,
        include_bias=False
    )

    model = ps.SINDy(
        feature_library=library,
        differentiation_method=ps.FiniteDifference(order=2),
        optimizer=ps.STLSQ(
            threshold=0.5,
            alpha=1e-8,
            normalize_columns=True
        )
    )

    # Time is supplied as an external input, not a state.
    model.fit(
        data,
        t=time,
        u=time.reshape(-1, 1),
        feature_names=["S", "R", "t"]
    )

    print("\nCandidate functions:")
    for name in model.get_feature_names():
        print(name)

    return model

def main() -> None:
    time, data = simulate_system()
    model = fit_sindy_model(time, data)

    print(f"Simulated {len(time)} time points from {time[0]:g} to {time[-1]:g}.")
    print(f"Initial state: S={data[0, 0]:.3f}, R={data[0, 1]:.3f}")
    print(f"Final state: S={data[-1, 0]:.3f}, R={data[-1, 1]:.3f}")
    print("\nSINDy model:")
    model.print()


if __name__ == "__main__":
    main()
