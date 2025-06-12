# ruff: noqa: E741

import time

import numpy as np
from interpatient import S_f_i
from q_learning import QLearningEtaOptimizer

from controller import control
from exercises import exercise_arrays
from meals import U_G_array
from metrics import calculate_metrics, exclude_training
from parameters import T, dt, dt_ratio, params, t0
from patient import Patient
from utils import is_multiple


def run_simulation(
    params: dict, optimizer: QLearningEtaOptimizer, patient_idx: int | None
) -> dict:
    if patient_idx is not None:
        patient = Patient(patient_idx)
    else:
        patient = Patient()

    print(patient.weight)
    times = np.arange(t0, T, dt, dtype=float)
    N_NEURONS = params["centers"].shape[0]

    S_1 = np.empty_like(times, dtype=float)
    S_2 = np.empty_like(times, dtype=float)
    I = np.empty_like(times, dtype=float)
    X_1 = np.empty_like(times, dtype=float)
    X_2 = np.empty_like(times, dtype=float)
    X_3 = np.empty_like(times, dtype=float)
    Q_1 = np.empty_like(times, dtype=float)
    Q_2 = np.empty_like(times, dtype=float)
    G = np.empty_like(times, dtype=float)
    G_s = np.empty_like(times, dtype=float)

    G_p = np.empty_like(times, dtype=float)
    d = np.empty_like(times, dtype=float)
    d_hat = np.empty_like(times, dtype=float)
    fbl = np.empty_like(times, dtype=float)

    u = np.empty_like(times, dtype=float)
    w = np.empty((times.shape[0], N_NEURONS), dtype=float)
    s = np.empty_like(times, dtype=float)

    S_1[0] = patient.S_1_0
    S_2[0] = patient.S_2_0
    I[0] = patient.I_0
    X_1[0] = patient.X_1_0
    X_2[0] = patient.X_2_0
    X_3[0] = patient.X_3_0
    Q_1[0] = patient.Q_1_0
    Q_2[0] = patient.Q_2_0
    G[0] = patient.G_0
    G_s[0] = patient.G_s_0

    params["eta"] = optimizer.run(times[0])

    s[0] = 0
    w[0] = params["w0"]
    u[0] = control(G_s[0], 0, w, d_hat, fbl, 0, times[0], s, params=params)

    U_G = U_G_array(times, patient.idx)
    M_PGU, M_PIU, M_HGP = exercise_arrays(times, patient.idx)

    for idx, t in enumerate(times[1:], 1):
        S_f1 = S_f_i(patient.S_f1, t)
        S_f2 = S_f_i(patient.S_f2, t)
        S_f3 = S_f_i(patient.S_f3, t)

        S_1_p = u[idx - 1] - S_1[idx - 1] / patient.t_max_I
        S_2_p = S_1[idx - 1] / patient.t_max_I - S_2[idx - 1] / patient.t_max_I
        I_p = (
            S_2[idx - 1] / (patient.t_max_I * patient.V_I)
            - patient.k_e * I[idx - 1]
        )
        X_1_p = (
            -patient.k_a1 * X_1[idx - 1]
            + M_PGU[idx - 1]
            * M_PIU[idx - 1]
            * S_f1
            * patient.k_a1
            * I[idx - 1]
        )
        X_2_p = (
            -patient.k_a2 * X_2[idx - 1]
            + M_PGU[idx - 1]
            * M_PIU[idx - 1]
            * S_f2
            * patient.k_a2
            * I[idx - 1]
        )
        X_3_p = (
            -patient.k_a3 * X_3[idx - 1]
            + M_HGP[idx - 1] * S_f3 * patient.k_a3 * I[idx - 1]
        )
        Q_1_p = (
            -X_1[idx - 1] * Q_1[idx - 1]
            + patient.k_12 * Q_2[idx - 1]
            - patient.F_C_01(G[idx - 1])
            - patient.F_R(G[idx - 1])
            + U_G[idx - 1]
            + patient.EGP(X_3[idx - 1])
        )
        Q_2_p = (
            X_1[idx - 1] * Q_1[idx - 1]
            - (patient.k_12 + X_2[idx - 1]) * Q_2[idx - 1]
        )
        G_s_p = (1 / patient.t_s) * (G[idx - 1] - G_s[idx - 1])
        G_p[idx - 1] = Q_1_p / patient.V_G
        d[idx - 1] = G_p[idx - 1] + u[idx - 1]

        S_1[idx] = S_1[idx - 1] + S_1_p * dt
        S_2[idx] = S_2[idx - 1] + S_2_p * dt
        I[idx] = I[idx - 1] + I_p * dt
        X_1[idx] = X_1[idx - 1] + X_1_p * dt
        X_2[idx] = X_2[idx - 1] + X_2_p * dt
        X_3[idx] = X_3[idx - 1] + X_3_p * dt
        Q_1[idx] = Q_1[idx - 1] + Q_1_p * dt
        Q_2[idx] = Q_2[idx - 1] + Q_2_p * dt
        G[idx] = Q_1[idx] / patient.V_G
        G_s[idx] = G_s[idx - 1] + G_s_p * dt

        params["eta"] = optimizer.run(t)

        G_d_p = 0
        G_d = params["G_d"] * patient.mg_dL_to_mmol_L
        s[idx] = (G_s_p - G_d_p) + params["lambda"] * (G_s[idx] - G_d)

        if is_multiple(idx, dt_ratio):
            u[idx] = control(
                G_s[idx], G_s_p, w, d_hat, fbl, idx, t, s, params=params
            )
        else:
            u[idx] = u[idx - 1]
            d_hat[idx] = d_hat[idx - 1]
            fbl[idx] = fbl[idx - 1]
            if idx + 1 < w.shape[0]:
                w[idx + 1] = w[idx]

    G_p[-1] = G_p[-2]
    d[-1] = d[-2]

    data = {
        "time": times / 60,
        "G": G * patient.mmol_L_to_mg_dL,
        "G_s": G_s * patient.mmol_L_to_mg_dL,
        "u": u * patient.mU_kg_min_to_U_h,
        "I": I,
        "U_G": U_G,
        "M_PGU": M_PGU,
        "M_PIU": M_PIU,
        "M_HGP": M_HGP,
        "w": w,
        "G_p": G_p,
        "d": d,
        "d_hat": d_hat,
        "fbl": fbl,
        "s": s * patient.mmol_L_to_mg_dL,
    }

    metrics = calculate_metrics(exclude_training(data))
    optimizer.update_q_table_episode(metrics)

    return data


def main(optimizer: QLearningEtaOptimizer, patient_idx: int | None) -> dict:
    start = time.perf_counter()

    params["eta"] = optimizer.current_eta
    data = run_simulation(params, optimizer, patient_idx)
    metrics = calculate_metrics(exclude_training(data))

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")

    return metrics


if __name__ == "__main__":
    main()
