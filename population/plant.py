# ruff: noqa: E741

import math

import numpy as np

DAYS = 3
T = DAYS * 24 * 60
t0 = 0
dt = 0.01

HOURS_STEADY_STATE = 3
idx_steady_state = round(HOURS_STEADY_STATE * 60 / dt)


meals = [
    {"day": 1, "hour": 8, "grams": 43.7},
    {"day": 1, "hour": 12, "grams": 43.7},
    {"day": 1, "hour": 18, "grams": 43.7},
    {"day": 2, "hour": 8, "grams": 43.7},
    {"day": 2, "hour": 12, "grams": 43.7},
    {"day": 2, "hour": 18, "grams": 43.7},
    {"day": 3, "hour": 8, "grams": 43.7},
    {"day": 3, "hour": 12, "grams": 43.7},
    {"day": 3, "hour": 18, "grams": 43.7},
]


def check_valid_patient(params: dict, with_insulin: bool) -> dict:
    times = np.arange(t0, T, dt, dtype=float)

    def U_G_array(times: np.ndarray) -> np.ndarray:
        threshold_effect = 10 * 60
        U_G = np.empty_like(times, dtype=float)
        for idx, t in enumerate(times):
            U_G_t = 0
            for meal in meals:
                meal_time = (meal["day"] - 1) * 24 * 60 + meal["hour"] * 60
                meal_cho = (
                    meal["grams"] * params["g_to_mmol"] / params["weight"]
                )
                if t >= meal_time and t < meal_time + threshold_effect:
                    U_G_meal = (
                        meal_cho
                        * params["A_G"]
                        * (t - meal_time)
                        * math.exp(-(t - meal_time) / params["t_max_G"])
                        / params["t_max_G"] ** 2
                    )
                    U_G_t += U_G_meal
            U_G[idx] = U_G_t
        return U_G

    S_1 = np.empty_like(times, dtype=float)
    S_2 = np.empty_like(times, dtype=float)
    I = np.empty_like(times, dtype=float)
    X_1 = np.empty_like(times, dtype=float)
    X_2 = np.empty_like(times, dtype=float)
    X_3 = np.empty_like(times, dtype=float)
    Q_1 = np.empty_like(times, dtype=float)
    Q_2 = np.empty_like(times, dtype=float)
    G = np.empty_like(times, dtype=float)

    initial = get_initial_conditions(params)
    S_1[0] = initial["S_1_0"]
    S_2[0] = initial["S_2_0"]
    I[0] = initial["I_0"]
    X_1[0] = initial["X_1_0"]
    X_2[0] = initial["X_2_0"]
    X_3[0] = initial["X_3_0"]
    Q_1[0] = initial["Q_1_0"]
    Q_2[0] = initial["Q_2_0"]
    G[0] = initial["G_0"]

    U_G = U_G_array(times)

    if with_insulin:
        mU_kg_min_to_U_h = 0.06 * params["weight"]
        u = 15 / mU_kg_min_to_U_h
    else:
        u = 0

    for idx, t in enumerate(times[1:], 1):
        S_1_p = u - S_1[idx - 1] / params["t_max_I"]
        S_2_p = (
            S_1[idx - 1] / params["t_max_I"] - S_2[idx - 1] / params["t_max_I"]
        )
        I_p = (
            S_2[idx - 1] / (params["t_max_I"] * params["V_I"])
            - params["k_e"] * I[idx - 1]
        )
        X_1_p = (
            -params["k_a1"] * X_1[idx - 1]
            + params["S_f1"] * params["k_a1"] * I[idx - 1]
        )
        X_2_p = (
            -params["k_a2"] * X_2[idx - 1]
            + params["S_f2"] * params["k_a2"] * I[idx - 1]
        )
        X_3_p = (
            -params["k_a3"] * X_3[idx - 1]
            + params["S_f3"] * params["k_a3"] * I[idx - 1]
        )
        Q_1_p = (
            -X_1[idx - 1] * Q_1[idx - 1]
            + params["k_12"] * Q_2[idx - 1]
            - F_C_01(G[idx - 1], params["F_01"])
            - F_R(G[idx - 1], params["V_G"])
            + U_G[idx - 1]
            + EGP(X_3[idx - 1], params["EGP_0"])
        )
        Q_2_p = (
            X_1[idx - 1] * Q_1[idx - 1]
            - (params["k_12"] + X_2[idx - 1]) * Q_2[idx - 1]
        )

        S_1[idx] = S_1[idx - 1] + S_1_p * dt
        S_2[idx] = S_2[idx - 1] + S_2_p * dt
        I[idx] = I[idx - 1] + I_p * dt
        X_1[idx] = X_1[idx - 1] + X_1_p * dt
        X_2[idx] = X_2[idx - 1] + X_2_p * dt
        X_3[idx] = X_3[idx - 1] + X_3_p * dt
        Q_1[idx] = Q_1[idx - 1] + Q_1_p * dt
        Q_2[idx] = Q_2[idx - 1] + Q_2_p * dt
        G[idx] = Q_1[idx] / params["V_G"]

        if idx > idx_steady_state:
            steady_state = G[(idx - idx_steady_state) : idx]
            G_with_insulin = 20 * params["mg_dL_to_mmol_L"]
            G_wo_insulin = 300 * params["mg_dL_to_mmol_L"]
            if with_insulin:
                if (steady_state < G_with_insulin).all():
                    return True
                else:
                    continue
            else:
                if (steady_state > G_wo_insulin).all():
                    return True
                else:
                    continue

    return False


def get_initial_conditions(params: dict) -> dict:
    mg_dL_to_mmol_L = 1 / 18
    I_b = 10
    G_b_mg_dL = 160
    G_b = G_b_mg_dL * mg_dL_to_mmol_L

    I_0 = I_b
    S_2_0 = params["k_e"] * I_b * params["t_max_I"] * params["V_I"]
    S_1_0 = S_2_0
    X_1_0 = params["S_f1"] * I_b
    X_2_0 = params["S_f2"] * I_b
    X_3_0 = params["S_f3"] * I_b
    G_0 = G_b
    Q_1_0 = G_0 * params["V_G"]
    Q_2_0 = (X_1_0 * Q_1_0) / (params["k_12"] + X_2_0)

    initial_conditions = {
        "I_0": I_0,
        "S_2_0": S_2_0,
        "S_1_0": S_1_0,
        "X_1_0": X_1_0,
        "X_2_0": X_2_0,
        "X_3_0": X_3_0,
        "G_0": G_0,
        "Q_1_0": Q_1_0,
        "Q_2_0": Q_2_0,
    }

    return initial_conditions


def F_C_01(G: float, F_01: float) -> float:
    if G >= 4.5:
        return F_01
    return F_01 * G / 4.5


def F_R(G: float, V_G: float) -> float:
    if G >= 9:
        return 0.003 * (G - 9) * V_G
    return 0.0


def EGP(X_3: float, EGP_0: float) -> float:
    return EGP_0 * (1 - X_3)
