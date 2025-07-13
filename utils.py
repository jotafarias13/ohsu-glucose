import polars as pl

import patient


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
