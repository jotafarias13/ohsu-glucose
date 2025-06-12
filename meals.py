import math

import numpy as np

from patient import Patient, PopulationType, get_patient


def U_G_array(
    times: np.ndarray, population_type: PopulationType, patient_idx: int
) -> np.ndarray:
    patient = Patient(population_type, patient_idx)
    meals = get_patient(population_type, patient_idx)["data"]["meals"]
    threshold_effect = 10 * 60

    U_G = np.empty_like(times, dtype=float)

    for idx, t in enumerate(times):
        U_G_t = 0

        for meal in meals:
            meal_time = (meal["day"] - 1) * 24 * 60 + meal["hour"] * 60
            meal_cho = meal["grams"] * patient.g_to_mmol / patient.weight

            if t >= meal_time and t < meal_time + threshold_effect:
                U_G_meal = (
                    meal_cho
                    * patient.A_G
                    * (t - meal_time)
                    * math.exp(-(t - meal_time) / patient.t_max_G)
                    / patient.t_max_G**2
                )
                U_G_t += U_G_meal

        U_G[idx] = U_G_t

    return U_G
