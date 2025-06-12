import math

import numpy as np

from patient import PopulationType, get_patient


def exercise_arrays(
    times: np.ndarray, population_type: PopulationType, patient_idx: int
) -> tuple[np.ndarray]:
    patient = get_patient(population_type, patient_idx)
    exercises = patient["data"]["exercises"]
    PAMM = 0.5

    M_PGU = np.empty_like(times, dtype=float)
    M_PIU = np.empty_like(times, dtype=float)
    M_HGP = np.empty_like(times, dtype=float)

    for idx, t in enumerate(times):
        for exercise in exercises:
            exercise_time = (exercise["day"] - 1) * 24 * 60 + exercise[
                "hour"
            ] * 60
            if t >= exercise_time and t < exercise_time + exercise["duration"]:
                PGUA_steady = (
                    0.006 * exercise["p_vo2_max"] ** 2
                    + 1.2264 * exercise["p_vo2_max"]
                    - 10.1958
                )
                PGUA = (
                    -PGUA_steady * math.exp(-(t - exercise_time) / 30)
                    + PGUA_steady
                )
                HGPA = PGUA

                M_PGU[idx] = 1 + PGUA * PAMM / 35
                M_PIU[idx] = 1 + 2.4 * PAMM
                M_HGP[idx] = 1 + HGPA * PAMM / 155
                break

        else:
            M_PGU[idx] = 1
            M_PIU[idx] = 1
            M_HGP[idx] = 1

    return M_PGU, M_PIU, M_HGP
