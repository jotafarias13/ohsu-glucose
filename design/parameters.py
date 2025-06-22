import json
import re
from enum import Enum
from pathlib import Path

import numpy as np

t0 = 0
dt = 0.01
dt_ctr = 5.0
dt_ratio = round(dt_ctr / dt)


class ControllerType(str, Enum):
    FBL = "FBL"
    FBL_RBF = "FBL_RBF"
    FBL_RBF_2 = "FBL_RBF_2"


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


def eta_fbl_rbf(t: float) -> float:
    """
    Day 1:  0.0015
    Day 2:  0.0010
    Day 3:  0.0005
    Day 4+: 0.0003
    """
    t_ = t / (60 * 24)
    if t_ < 1:
        return 0.0015
    if t_ < 2:
        return 0.0010
    if t_ < 3:
        return 0.0005
    return 0.0003


centers_1 = np.array([0, 2, 4, 6, 8, 10, 12, 14])
widths_1 = np.array([1.0] * len(centers_1))

centers_2 = np.array([0.05, 0.15, 0.25, 0.35])
widths_2 = np.array([0.05] * len(centers_2))

params = [
    {
        "controller_type": ControllerType.FBL,
        "days": 7,
        "days_training": 3,
        "T": 7 * 24 * 60,
        "lambda": 0.08,
        "centers": np.array([]),
        "widths": np.array([]),
        "eta": 0,
        "w_max": 0,
        "G_d": 90,
        "w0": [],
    },
    {
        "controller_type": ControllerType.FBL_RBF,
        "days": 7,
        "days_training": 3,
        "T": 7 * 24 * 60,
        "lambda": 0.005,
        "centers": centers_1,
        "widths": widths_1,
        "eta": eta_fbl_rbf,
        "w_max": 0.5,
        "G_d": 90,
        "w0": [0] * len(centers_1),
    },
    {
        "controller_type": ControllerType.FBL_RBF_2,
        "days": 7,
        "days_training": 3,
        "T": 7 * 24 * 60,
        "lambda": 0.015,
        "centers": centers_2,
        "widths": widths_2,
        "eta": 0.002,
        "w_max": 0.8,
        "G_d": 90,
        "w0": [0] * len(centers_2),
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
