# ruff: noqa: E741

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import graphs
from controller import control
from exercises import exercise_arrays
from meals import U_G_array
from metrics import (
    calculate_metrics,
    exclude_training,
    print_metrics,
    save_metrics,
)
from parameters import (
    T,
    dt,
    dt_ctr,
    dt_ratio,
    params,
    params_offline,
    params_test,
    params_train,
    save_params,
    t0,
)
from patient import Patient, PopulationType
from population.interpatient import S_f_i
from utils import is_multiple, process_args, save_data


def run_simulation(params: dict) -> dict:
    patient = Patient(params["population_type"], params["patient_idx"])

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
    G_s_pp_est = np.empty_like(times, dtype=float)
    d = np.empty_like(times, dtype=float)
    d_hat = np.empty_like(times, dtype=float)
    fbl = np.empty_like(times, dtype=float)

    u = np.empty_like(times, dtype=float)
    w = np.empty((times.shape[0], N_NEURONS), dtype=float)
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

    d_hat[0] = 0
    G_s_p_est[0] = 0
    s[0] = 0
    G_p_error[0] = 0
    G_s_pp_est[0] = 0
    w[0] = params["w0"]
    u[0] = control(
        G_s[0], G_s_p_est[0], w, d_hat, fbl, 0, times[0], s, params=params
    )

    U_G = U_G_array(times, params["population_type"], patient.idx)
    M_PGU, M_PIU, M_HGP = exercise_arrays(
        times, params["population_type"], patient.idx
    )

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
            u[idx] = control(
                G_s[idx],
                G_s_p_est[idx],
                w,
                d_hat,
                fbl,
                idx,
                t,
                s,
                params=params,
            )
            G_s_pp_est[idx] = (G_s_p_est[idx] - G_s_p_est[prev]) / dt_ctr

        else:
            G_s_p_est[idx] = G_s_p_est[idx - 1]
            s[idx] = s[idx - 1]
            u[idx] = u[idx - 1]
            G_p_error[idx] = G_p_error[idx - 1]
            d_hat[idx] = d_hat[idx - 1]
            fbl[idx] = fbl[idx - 1]
            if idx + 1 < w.shape[0]:
                w[idx + 1] = w[idx]
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
        "M_PGU": M_PGU,
        "M_PIU": M_PIU,
        "M_HGP": M_HGP,
        "w": w,
        "G_p": G_p * patient.mmol_L_to_mg_dL,
        "d": d,
        "d_hat": d_hat,
        "fbl": fbl,
        "s": s,
        "G_p_error": G_p_error,
        "G_error": G_error,
        "l_G_error": l_G_error,
        "G_s_pp": G_s_pp_est * patient.mmol_L_to_mg_dL,
    }

    return data


def run_graphs(data: dict, params: dict) -> dict:
    graphs.run(data, params)
    metrics = calculate_metrics(exclude_training(data))
    save_metrics(metrics, params["graphs_dir"])
    save_params(params)

    return metrics


def run(params: dict, *, save=bool) -> dict:
    data = run_simulation(params)
    if save:
        save_data(data, params)
    metrics = run_graphs(data, params)

    return metrics


def main() -> None:
    start = time.perf_counter()
    args = process_args()
    population = args["population"]
    save = args["save"]
    parallel = args["parallel"]

    if population == PopulationType.TRAIN:
        results_dir = Path.cwd() / "results_train"
        params_list = params_train
    elif population == PopulationType.TEST:
        results_dir = Path.cwd() / "results_test"
        params_list = params_test
    elif population == PopulationType.OFFLINE:
        results_dir = Path.cwd() / "results_offline"
        params_list = params_offline
    else:
        results_dir = Path.cwd() / "results"
        params_list = params
    results_dir.mkdir(exist_ok=True)

    for idx, params_ in enumerate(params_list):
        params_["graphs_dir"] = results_dir / f"sim_{idx}"

    if parallel:
        cpus = min(mp.cpu_count(), 5)
        with mp.Pool(processes=cpus) as pool:
            metrics_list = pool.map(run, params_list, save)
    else:
        metrics_list = [run(params_list[0], save=save)]

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")

    for metrics in metrics_list:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
