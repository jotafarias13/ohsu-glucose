DAYS = 1
DAYS_TRAINING = 0
T = DAYS * 24 * 60
t0 = 0
dt = 0.01
dt_ctr = 5.0
dt_ratio = round(dt_ctr / dt)


params = [
    {"patient_idx": 0, "lambda": 0.015, "G_d": 90},
    {"patient_idx": 1, "lambda": 0.015, "G_d": 90},
    {"patient_idx": 2, "lambda": 0.015, "G_d": 90},
    {"patient_idx": 3, "lambda": 0.015, "G_d": 90},
    {"patient_idx": 4, "lambda": 0.015, "G_d": 90},
]
