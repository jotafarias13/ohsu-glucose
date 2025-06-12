import json
import math
import re
from pathlib import Path

import numpy as np

from patient import PopulationType

DAYS = 10
DAYS_TRAINING = 5
T = DAYS * 24 * 60
t0 = 0
dt = 0.01
dt_ctr = 5.0
dt_ratio = round(dt_ctr / dt)


def eta_offline(t: float) -> float:
    """\
    Day 1:  0.0022500
    Day 2:  0.0020250
    Day 3:  0.0002025
    Day 4+: 0.0002025\
    """
    t_ = t / (60 * 24)
    if t_ < 1:
        return 0.00225
    elif t_ < 2:
        return 0.002025
    elif t_ < 3:
        return 0.0002025
    else:
        return 0.0002025


def eta_rl(t: float) -> float:
    """
    Day 1:  0.0025000
    Day 2:  0.0022500
    Day 3:  0.0020250
    Day 4+: 0.0002025
    """
    t_ = t / (60 * 24)
    if t_ < 1:
        return 0.0025
    elif t_ < 2:
        return 0.00225
    elif t_ < 3:
        return 0.002025
    else:
        return 0.0002025


centers = np.array([0.05, 0.15, 0.25, 0.35])
widths = np.array([0.05] * len(centers))

params = [
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 0,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 1,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 2,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 3,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 3,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_offline,
        "w_max": 0.5,
        "G_d": 90,
        "w0": [0.25694895, 0.35747646, 0.20557538, -0.02384593],
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 4,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.7,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
]


def save_params(params: dict) -> None:
    params_file = params["graphs_dir"] / "params.json"

    params_ = params.copy()
    params_["centers"] = params_["centers"].tolist()
    params_["widths"] = params_["widths"].tolist()
    if callable(params_["eta"]):
        params_["eta"] = re.sub(
            r"\s+", " ", params["eta"].__doc__.strip("\n").replace("\n", " | ")
        )

    params_.pop("graphs_dir")

    with Path.open(params_file, "w") as file:
        json.dump(params_, file, indent=4, ensure_ascii=False)


params_train = [
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 0,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 1,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.7,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 2,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.9,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 3,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.65,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 4,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.65,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 5,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.7,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 6,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.7,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 7,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.65,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 8,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.7,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TRAIN,
        "patient_idx": 9,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.5,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
]


params_test = [
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 0,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 1,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 2,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 3,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 0.6,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 4,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_rl,
        "w_max": 1.0,
        "G_d": 90,
        "w0": [0] * len(centers),
    },
]


def get_offline_weights() -> dict:
    offline_data = {}
    weights_dir = Path.cwd() / "offline" / "weights"
    for patient_idx in range(5):
        weights_file = weights_dir / f"weights_sim_{patient_idx}.json"
        with Path.open(weights_file, "r") as file:
            weights_sim = json.load(file)
        offline_data[f"sim_{patient_idx}"] = weights_sim

    return offline_data


def correct_norm(weights_norm: float) -> float:
    return math.ceil(weights_norm * 10) / 10


offline_weigths = get_offline_weights()

params_offline = [
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 0,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_offline,
        "w_max": correct_norm(offline_weigths["sim_0"]["weights_norm"]),
        "G_d": 90,
        "w0": offline_weigths["sim_0"]["weights"],
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 1,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_offline,
        "w_max": correct_norm(offline_weigths["sim_1"]["weights_norm"]),
        "G_d": 90,
        "w0": offline_weigths["sim_1"]["weights"],
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 2,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_offline,
        "w_max": correct_norm(offline_weigths["sim_2"]["weights_norm"]),
        "G_d": 90,
        "w0": offline_weigths["sim_2"]["weights"],
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 3,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_offline,
        "w_max": correct_norm(offline_weigths["sim_3"]["weights_norm"]),
        "G_d": 90,
        "w0": offline_weigths["sim_3"]["weights"],
    },
    {
        "population_type": PopulationType.TEST,
        "patient_idx": 4,
        "lambda": 0.015,
        "centers": centers,
        "widths": widths,
        "eta": eta_offline,
        "w_max": correct_norm(offline_weigths["sim_4"]["weights_norm"]),
        "G_d": 90,
        "w0": offline_weigths["sim_4"]["weights"],
    },
]
