import json
from pathlib import Path

import numpy as np
import polars as pl


def calculate_weigths(patient_idx: int) -> dict:
    file = f"results/sim_{patient_idx}/data.parquet"
    data = pl.read_parquet(file, columns=["uncertainty", "s"])

    centers = np.array([0.05, 0.15, 0.25, 0.35])
    lengths = np.array([0.05] * len(centers))

    def activation(
        error: float, centers: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        phi = np.exp(-(1.0 / 2.0) * np.square((error - centers) / lengths))
        return phi

    Gc = [activation(center, centers, lengths) for center in centers]
    Gc = np.array(Gc)

    G = [activation(error, centers, lengths) for error in data["s"]]
    G = np.array(G)

    phi = 100.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Gp = np.linalg.inv(G.T @ G + phi * Gc)
        Gp = Gp @ G.T

    d = np.array(data["uncertainty"])
    w = Gp @ d
    w_norm = np.linalg.norm(w)

    return {"weights": w.tolist(), "weights_norm": float(w_norm)}


def main() -> None:
    weights_dir = Path.cwd() / "weights"
    weights_dir.mkdir(exist_ok=True)

    for patient_idx in range(5):
        w = calculate_weigths(patient_idx)
        weights_file = weights_dir / f"weights_sim_{patient_idx}.json"
        with Path.open(weights_file, "w") as file:
            json.dump(w, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
