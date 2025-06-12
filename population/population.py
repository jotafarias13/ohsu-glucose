import json
import random
from pathlib import Path

import numpy as np
from plant import check_valid_patient

POPULATION_SIZE = 15
MEALS_PER_DAY = 3
EXERCISES_PER_DAY = 1
DAYS_PER_SIMULATION = 10


def generate_population() -> list[dict]:
    size = POPULATION_SIZE

    def generate_weights() -> np.ndarray:
        weight_mean, weight_std = 76.3, 14.6
        weights = np.random.normal(weight_mean, weight_std, size)
        weights = np.round(weights).astype(int)
        if weights.min() < weight_mean - 1.5 * weight_std:
            return generate_weights()
        if weights.max() > weight_mean + 1.5 * weight_std:
            return generate_weights()
        return weights

    weights = generate_weights()

    def generate_sf1() -> np.ndarray:
        s_f_1_mean, s_f_1_std = 21e-4, 5.9e-4
        s_f_1s = np.random.normal(s_f_1_mean, s_f_1_std, size)
        if s_f_1s.min() < s_f_1_mean - 1.0 * s_f_1_std:
            return generate_sf1()
        if s_f_1s.max() > s_f_1_mean + 1.0 * s_f_1_std:
            return generate_sf1()
        return s_f_1s

    s_f_1s = generate_sf1()

    def generate_sf2() -> np.ndarray:
        s_f_2_mean, s_f_2_std = 3.5e-4, 1.4e-4
        s_f_2s = np.random.normal(s_f_2_mean, s_f_2_std, size)
        if s_f_2s.min() < s_f_2_mean - 1.0 * s_f_2_std:
            return generate_sf2()
        if s_f_2s.max() > s_f_2_mean + 1.0 * s_f_2_std:
            return generate_sf2()
        return s_f_2s

    s_f_2s = generate_sf2()

    def generate_sf3() -> np.ndarray:
        s_f_3_mean, s_f_3_std = 214e-4, 5.9e-4
        s_f_3s = np.random.normal(s_f_3_mean, s_f_3_std, size)
        if s_f_3s.min() < s_f_3_mean - 1.0 * s_f_3_std:
            return generate_sf3()
        if s_f_3s.max() > s_f_3_mean + 1.0 * s_f_3_std:
            return generate_sf3()
        return s_f_3s

    s_f_3s = generate_sf3()

    population = []
    for weight, sf1, sf2, sf3 in zip(weights, s_f_1s, s_f_2s, s_f_3s):
        patient = {
            "weight": int(weight),
            "mmol_L_to_mg_dL": 18,
            "mg_dL_to_mmol_L": 1 / 18,
            "g_to_mmol": 100 / 18,
            "mU_kg_min_to_U_h": 0.06 * float(weight),
            "t_max_I": 55,
            "V_I": 0.12,
            "k_e": 0.138,
            "k_a1": 0.006,
            "k_a2": 0.06,
            "k_a3": 0.03,
            "S_f1": float(sf1),
            "S_f2": float(sf2),
            "S_f3": float(sf3),
            "k_12": 0.066,
            "F_01": 0.0097,
            "EGP_0": 0.0161,
            "A_G": 0.8,
            "t_max_G": 40,
            "V_G": 0.16,
        }
        population.append({"patient": patient})

    for p in population:
        valid_insulin = check_valid_patient(p["patient"], with_insulin=True)
        valid_no_insulin = check_valid_patient(
            p["patient"], with_insulin=False
        )
        if not valid_insulin or not valid_no_insulin:
            print("Generating new population...")
            return generate_population()

    return population


def generate_meals(population: list[dict]) -> list[dict]:
    meal_days = list(range(1, DAYS_PER_SIMULATION + 1))
    days = []
    for day in meal_days:
        days += [day] * MEALS_PER_DAY

    def generate_meal_chos() -> dict:
        breakfast = {"mean": 32.2, "std": 11.1}
        lunch = {"mean": 57.2, "std": 15.9}
        dinner = {"mean": 40.2, "std": 9.9}

        chos_breakfast = np.random.normal(
            breakfast["mean"], breakfast["std"], DAYS_PER_SIMULATION
        )
        chos_lunch = np.random.normal(
            lunch["mean"], lunch["std"], DAYS_PER_SIMULATION
        )
        chos_dinner = np.random.normal(
            dinner["mean"], dinner["std"], DAYS_PER_SIMULATION
        )

        if chos_breakfast.min() < breakfast["mean"] - 1.5 * breakfast["std"]:
            return generate_meal_chos()
        if chos_breakfast.max() > breakfast["mean"] + 1.5 * breakfast["std"]:
            return generate_meal_chos()

        if chos_lunch.min() < lunch["mean"] - 1.5 * lunch["std"]:
            return generate_meal_chos()
        if chos_lunch.max() > lunch["mean"] + 1.5 * lunch["std"]:
            return generate_meal_chos()

        if chos_dinner.min() < dinner["mean"] - 1.5 * dinner["std"]:
            return generate_meal_chos()
        if chos_dinner.max() > dinner["mean"] + 1.5 * dinner["std"]:
            return generate_meal_chos()

        chos = np.column_stack((
            chos_breakfast,
            chos_lunch,
            chos_dinner,
        )).ravel()
        chos = np.round(chos).astype(int)

        return chos

    def generate_meal_times() -> dict:
        hours_list = [[6, 7, 8], [12, 13, 14], [18, 19, 20, 21]]
        hours_breakfast = np.random.choice(
            hours_list[0], size=DAYS_PER_SIMULATION, replace=True
        )
        hours_lunch = np.random.choice(
            hours_list[1], size=DAYS_PER_SIMULATION, replace=True
        )
        hours_dinner = np.random.choice(
            hours_list[2], size=DAYS_PER_SIMULATION, replace=True
        )
        hours = np.column_stack((
            hours_breakfast,
            hours_lunch,
            hours_dinner,
        )).ravel()
        return hours

    for patient in population:
        hours = generate_meal_times()
        grams = generate_meal_chos()
        meals = []
        for day, hour, gram in zip(days, hours, grams):
            meal = {"day": day, "hour": int(hour), "grams": int(gram)}
            meals.append(meal)
        patient["meals"] = meals

    return population


def generate_exercises() -> list[dict]:
    size = DAYS_PER_SIMULATION
    vo2_max_mean = 60
    vo2_max_std = 10
    p_vo2_maxs = np.random.normal(vo2_max_mean, vo2_max_std, size)
    p_vo2_maxs = np.round(p_vo2_maxs).astype(int)
    if p_vo2_maxs.min() < vo2_max_mean - 1.5 * vo2_max_std:
        return generate_exercises()
    if p_vo2_maxs.max() > vo2_max_mean + 1.5 * vo2_max_std:
        return generate_exercises()

    duration_mean = 45
    duration_std = 10
    durations = np.random.normal(duration_mean, duration_std, size)
    durations = np.round(durations).astype(int)
    if durations.min() < duration_mean - 1.5 * duration_std:
        return generate_exercises()
    if durations.max() > duration_mean + 1.5 * duration_std:
        return generate_exercises()

    hours_list = [6, 7, 11, 12, 13, 18, 19, 20]
    hours = np.random.choice(hours_list, size=size, replace=True)

    exercises = []
    for idx in range(size):
        exercise = {
            "day": idx + 1,
            "hour": int(hours[idx]),
            "p_vo2_max": int(p_vo2_maxs[idx]),
            "duration": int(durations[idx]),
        }
        exercises.append(exercise)

    return exercises


def save_population(population: list[dict]) -> None:
    half_population = 10
    train = population[:half_population]
    test = population[half_population:]

    with Path.open("population/population_train.json", "w") as file:
        json.dump(train, file, indent=4, ensure_ascii=False)

    with Path.open("population/population_test.json", "w") as file:
        json.dump(test, file, indent=4, ensure_ascii=False)


def main() -> None:
    population = generate_population()
    population = generate_meals(population)
    for patient in population:
        patient["exercises"] = generate_exercises()

    random.shuffle(population)
    save_population(population)


if __name__ == "__main__":
    main()
