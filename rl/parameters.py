import numpy as np

DAYS = 10
DAYS_TRAINING = 5
T = DAYS * 24 * 60
t0 = 0
dt = 0.01
dt_ctr = 5.0
dt_ratio = round(dt_ctr / dt)


centers = np.array([0.05, 0.15, 0.25, 0.35])
widths = np.array([0.05] * len(centers))

params = {
    "lambda": 0.015,
    "centers": centers,
    "widths": widths,
    "eta": None,
    "w_max": 0.8,
    "G_d": 90,
    "w0": [0] * len(centers),
}
