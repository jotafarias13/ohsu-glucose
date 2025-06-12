# ruff: noqa: E741

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from interpatient import S_f_i

import graphs
from meals import U_G_array
from parameters import T, dt, dt_ctr, dt_ratio, params, t0
from patient import Patient
from utils import is_multiple, save_data


def run_simulation(params: dict) -> dict:
    patient = Patient(params["patient_idx"])

    times = np.arange(t0, T, dt, dtype=float)

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
    G_s_pp_est = np.empty_like(times, dtype=float)
    d = np.empty_like(times, dtype=float)

    u = np.empty_like(times, dtype=float)
    G_s_p_est = np.empty_like(times, dtype=float)
    s = np.empty_like(times, dtype=float)
    G_p_error = np.empty_like(times, dtype=float)

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

    G_s_p_est[0] = 0
    s[0] = 0
    G_p_error[0] = 0
    G_s_pp_est[0] = 0
    u[0] = 0

    U_G = U_G_array(times, patient.idx)

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
            + 1.0 * 1.0 * S_f1 * patient.k_a1 * I[idx - 1]
        )
        X_2_p = (
            -patient.k_a2 * X_2[idx - 1]
            + 1.0 * 1.0 * S_f2 * patient.k_a2 * I[idx - 1]
        )
        X_3_p = (
            -patient.k_a3 * X_3[idx - 1]
            + 1.0 * S_f3 * patient.k_a3 * I[idx - 1]
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

        G_d_p = 0
        G_d = params["G_d"] * patient.mg_dL_to_mmol_L

        if is_multiple(idx, dt_ratio):
            prev = idx - dt_ratio
            G_s_p_est[idx] = (G_s[idx] - G_s[prev]) / dt_ctr
            s[idx] = (G_s_p_est[idx] - G_d_p) + params["lambda"] * (
                G_s[idx] - G_d
            )
            G_p_error[idx] = G_s_p_est[idx] - G_d_p
            Gx = G_s[idx] * patient.mmol_L_to_mg_dL
            if Gx < 70:
                ux = 0
            elif Gx < 100:
                ux = 0.0
            elif Gx < 120:
                ux = 0.0
            elif Gx < 180:
                ux = 2.0
            else:
                ux = 2.0
            u[idx] = ux / patient.mU_kg_min_to_U_h
            G_s_pp_est[idx] = (G_s_p_est[idx] - G_s_p_est[prev]) / dt_ctr

        else:
            G_s_p_est[idx] = G_s_p_est[idx - 1]
            s[idx] = s[idx - 1]
            u[idx] = u[idx - 1]
            G_p_error[idx] = G_p_error[idx - 1]
            G_s_pp_est[idx] = G_s_pp_est[idx - 1]

        d[idx - 1] = G_s_pp_est[idx - 1] + u[idx - 1]

    G_p[-1] = G_p[-2]
    d[-1] = d[-2]

    l_G_error = s - G_p_error
    G_error = l_G_error / params["lambda"]

    data = {
        "time": times / 60,
        "G": G * patient.mmol_L_to_mg_dL,
        "G_s": G_s * patient.mmol_L_to_mg_dL,
        "u": u * patient.mU_kg_min_to_U_h,
        "I": I,
        "U_G": U_G,
        "G_p": G_p * patient.mmol_L_to_mg_dL,
        "d": d,
        "s": s,
        "G_p_error": G_p_error,
        "G_error": G_error,
        "l_G_error": l_G_error,
        "G_s_pp": G_s_pp_est * patient.mmol_L_to_mg_dL,
    }

    return data


def run(params: dict) -> None:
    data = run_simulation(params)
    save_data(data, params)
    graphs.run(data, params)


def main() -> None:
    start = time.perf_counter()

    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)

    for idx, params_ in enumerate(params):
        params_["graphs_dir"] = results_dir / f"sim_{idx}"

    with mp.Pool(processes=5) as pool:
        _ = pool.map(run, params)

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
