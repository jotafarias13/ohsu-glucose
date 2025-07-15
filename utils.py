import argparse

import polars as pl

import patient
from patient import PopulationType


def save_data(data: dict, params: dict) -> None:
    graphs_dir = params["graphs_dir"]
    graphs_dir.mkdir(exist_ok=True)

    data_ = pl.DataFrame(data)
    data_ = data_.select(
        pl.col("d").alias("uncertainty"),
        ((pl.col("G_s") - pl.lit(90)) * patient.mg_dL_to_mmol_L).alias(
            "error"
        ),
        pl.col("u"),
        pl.col("time"),
        pl.col("d_hat"),
        pl.col("s"),
        pl.col("G"),
        pl.col("G_p"),
    )
    data_file = "data.parquet"
    data_path = graphs_dir / data_file
    data_.write_parquet(data_path)


def is_multiple(large: int, small: int) -> bool:
    return large % small == 0


def process_args() -> dict:
    parser = argparse.ArgumentParser(
        prog="Run simulation for virtual patients",
    )

    parser.add_argument(
        "-p",
        "--population",
        type=str,
        default="normal",
        choices=[p.value.lower() for p in list(PopulationType)],
        help="Population to run rimulations on",
    )

    parser.add_argument(
        "-s",
        "--not-save",
        action="store_false",
        help="Whether to save the simulation data",
    )

    parser.add_argument(
        "-l",
        "--not-parallel",
        action="store_false",
        help="Whether to run the simulation in parallel",
    )

    args = parser.parse_args()
    population = PopulationType(args.population.upper())
    save = args.not_save
    parallel = args.not_parallel

    return {"population": population, "save": save, "parallel": parallel}
