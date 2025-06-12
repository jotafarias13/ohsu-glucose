import json
import random
from enum import Enum
from pathlib import Path

# Conversion factors
# glucose concentration from [mmol L-1] to [mg dL-1]
mmol_L_to_mg_dL = 18  # noqa: N816
# glucose concentration from [mg dL-1] to [mmol L-1]
mg_dL_to_mmol_L = 1 / mmol_L_to_mg_dL  # noqa: N816


class PopulationType(str, Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"


def get_patient(
    population_type: PopulationType, idx: int | None = None
) -> dict:
    if population_type == PopulationType.TRAIN:
        with Path.open("population/population_train.json", "r") as file:
            population = json.load(file)
    elif population_type == PopulationType.TEST:
        with Path.open("population/population_test.json", "r") as file:
            population = json.load(file)
    else:
        raise ValueError("Invalid population type")

    if idx is not None:
        patient_idx = idx
    else:
        patient_idx = random.choice(range(len(population)))

    return {"idx": patient_idx, "data": population[patient_idx]}


class Patient:
    def __init__(
        self, population_type: PopulationType, idx: int | None = None
    ) -> "Patient":
        pat = get_patient(population_type, idx)
        patient = pat["data"]["params"]
        self.idx = pat["idx"]
        self.weight = patient["weight"]
        self.mmol_L_to_mg_dL = patient["mmol_L_to_mg_dL"]
        self.mg_dL_to_mmol_L = 1.0 / self.mmol_L_to_mg_dL
        self.g_to_mmol = 100.0 / 18.0
        self.mU_kg_min_to_U_h = 0.06 * self.weight
        self.t_max_I = patient["t_max_I"]
        self.V_I = patient["V_I"]
        self.k_e = patient["k_e"]
        self.k_a1 = patient["k_a1"]
        self.k_a2 = patient["k_a2"]
        self.k_a3 = patient["k_a3"]
        self.S_f1 = patient["S_f1"]
        self.S_f2 = patient["S_f2"]
        self.S_f3 = patient["S_f3"]
        self.k_12 = patient["k_12"]
        self.F_01 = patient["F_01"]
        self.EGP_0 = patient["EGP_0"]
        self.A_G = patient["A_G"]
        self.t_max_G = patient["t_max_G"]
        self.V_G = patient["V_G"]
        self.t_s = 16
        self.G_b_mg_dL = 100
        self.G_b = self.G_b_mg_dL * self.mg_dL_to_mmol_L
        self.I_b = 10

        self.I_0 = self.I_b
        self.S_2_0 = self.k_e * self.I_b * self.t_max_I * self.V_I
        self.S_1_0 = self.S_2_0
        self.X_1_0 = self.S_f1 * self.I_b
        self.X_2_0 = self.S_f2 * self.I_b
        self.X_3_0 = self.S_f3 * self.I_b
        self.G_0 = self.G_b
        self.Q_1_0 = self.G_0 * self.V_G
        self.Q_2_0 = (self.X_1_0 * self.Q_1_0) / (self.k_12 + self.X_2_0)
        self.G_s_0 = self.G_0 * 0.95

    def F_C_01(self, G: float) -> float:
        if G >= 4.5:
            return self.F_01
        return self.F_01 * G / 4.5

    def F_R(self, G: float) -> float:
        if G >= 9:
            return 0.003 * (G - 9) * self.V_G
        return 0.0

    def EGP(self, X_3: float) -> float:
        return self.EGP_0 * (1 - X_3)
