from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from patient import PopulationType


def setup_matplotlib_params() -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"


def plot_aggregate(
    graphs_dir: Path, var: str, population: PopulationType
) -> None:
    if population == PopulationType.TRAIN:
        population_dir = Path.cwd() / "results_train"
        pop_size = 10
    if population == PopulationType.TEST:
        population_dir = Path.cwd() / "results_test"
        pop_size = 5

    data_file = population_dir / "sim_0" / "data.parquet"
    times = (
        pl.read_parquet(data_file, columns=["time"])
        .gather_every(100)
        .to_series()
    )
    patients = None

    for patient in range(pop_size):
        data_file = population_dir / f"sim_{patient}" / "data.parquet"
        data = (
            pl.read_parquet(data_file, columns=[var])
            .gather_every(100)
            .select(pl.col(var).alias(f"{var}{patient}"))
        )

        if patient == 0:
            patients = data
        else:
            patients = pl.concat([patients, data], how="horizontal")

    patients = patients.with_columns(
        patients.min_horizontal().alias("min"),
        patients.max_horizontal().alias("max"),
        patients.mean_horizontal().alias("mean"),
        pl.concat_list([f"{var}{i}" for i in range(pop_size)])
        .list.std(ddof=1)
        .alias("std"),
    )
    patients = patients.with_columns(
        (pl.col("mean") - pl.col("std")).alias("low"),
        (pl.col("mean") + pl.col("std")).alias("high"),
    )

    _, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.plot(
        times, patients["mean"], linewidth=2, color="#36A2EB", label="Mean"
    )
    ax.fill_between(
        times,
        patients["low"],
        patients["high"],
        color="skyblue",
        alpha=0.5,
        label="Standard Deviation",
    )

    ax.set_xlabel("Time", labelpad=20, fontsize=20)
    ax.set_xlim(left=0, right=240)

    ax.set_xticks(
        ticks=list(range(12, 10 * 24 + 1, 24)),
        labels=[f"Day {i}" for i in range(1, 11)],
        fontsize=16,
    )

    if var == "G":
        ax.set_ylim(0, 450)
        ax.set_ylabel(
            (
                "Glycemia Variability "
                r"$\left[ \si{mg {\cdot} dL^{-1}} \right]$"
            ),
            labelpad=20,
            fontsize=20,
        )
        ax.set_yticks(
            ticks=list(range(0, 301, 50)),
            labels=list(range(0, 301, 50)),
            fontsize=16,
        )
        ax.axhline(
            y=54, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax.axhline(
            y=70, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax.axhline(
            y=180, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax.axhline(
            y=250, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax.text(x=1, y=45, s="Severe Hypoglycemia", fontsize=12)
        ax.text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=12)
        ax.text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=12)
        ax.text(x=1, y=255, s="Severe Hyperglycemia", fontsize=12)
    if var == "u":
        ax.set_ylim(0, 4)
        ax.set_ylabel(
            (
                "Insulin Infusion Variability "
                r"$\left[ \si{U {\cdot} h^{-1}} \right]$"
            ),
            labelpad=20,
            fontsize=20,
        )

    ax.legend(prop={"size": 16})

    output_file = graphs_dir / f"{var}-{population.value}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_abnormal_glycemia(graphs_dir: Path) -> None:
    population_dir = Path.cwd() / "results_train"
    data_file = population_dir / "sim_4" / "data.parquet"
    data_5 = pl.read_parquet(data_file, columns=["time", "G"]).gather_every(
        100
    )
    data_file = population_dir / "sim_7" / "data.parquet"
    data_8 = pl.read_parquet(data_file, columns=["time", "G"]).gather_every(
        100
    )

    _, ax = plt.subplots(2, 1, figsize=(16, 12))
    ax[0].plot(
        data_5["time"],
        data_5["G"],
        linewidth=4,
        color="#36A2EB",
        label="G(t)",
    )

    ax[1].plot(
        data_8["time"],
        data_8["G"],
        linewidth=4,
        color="#36A2EB",
        label="G(t)",
    )

    pvo2 = r"$\text{PVO}_{2\,\max}$"
    ax[0].axvline(x=(6 - 1) * 24 + 18, color="#C9C9C9", linewidth=2)
    ax[0].axvline(x=(8 - 1) * 24 + 19, color="#C9C9C9", linewidth=2)
    ax[0].axvline(x=(8 - 1) * 24 + 14, color="#C9C9C9", linewidth=2)
    ax[0].text(x=(6 - 1) * 24 + 18 + 1, y=350, s="Exercise", fontsize=12)
    ax[0].text(x=(6 - 1) * 24 + 18 + 1, y=330, s=rf"68\% {pvo2}", fontsize=12)
    ax[0].text(x=(6 - 1) * 24 + 18 + 1, y=310, s="54 minutes", fontsize=12)
    ax[0].text(x=(8 - 1) * 24 + 19 + 1, y=350, s="Exercise", fontsize=12)
    ax[0].text(x=(8 - 1) * 24 + 19 + 1, y=330, s=rf"59\% {pvo2}", fontsize=12)
    ax[0].text(x=(8 - 1) * 24 + 19 + 1, y=310, s="42 minutes", fontsize=12)
    ax[0].text(x=(8 - 1) * 24 + 14 - 7.5, y=380, s="Meal", fontsize=12)
    ax[0].text(x=(8 - 1) * 24 + 14 - 13, y=360, s="75 grams", fontsize=12)

    ax[1].axvline(x=(8 - 1) * 24 + 18, color="#C9C9C9", linewidth=2)
    ax[1].text(x=(8 - 1) * 24 + 18 + 1, y=350, s="Exercise", fontsize=12)
    ax[1].text(
        x=(8 - 1) * 24 + 18 + 1,
        y=330,
        s=rf"73\% {pvo2}",
        fontsize=12,
    )
    ax[1].text(x=(8 - 1) * 24 + 18 + 1, y=310, s="40 minutes", fontsize=12)

    for i in range(2):
        ax[i].set_xlabel("Time", labelpad=20, fontsize=20)
        ax[i].set_xlim(left=0, right=240)

        ax[i].set_xticks(
            ticks=list(range(12, 10 * 24 + 1, 24)),
            labels=[f"Day {i}" for i in range(1, 11)],
            fontsize=16,
        )

        ax[i].set_ylim(0, 450)
        ax[i].set_ylabel(
            (
                "Blood Glucose Concentration "
                r"$\left[ \si{mg {\cdot} dL^{-1}} \right]$"
            ),
            labelpad=20,
            fontsize=18,
        )
        ax[i].set_yticks(
            ticks=list(range(0, 301, 50)),
            labels=list(range(0, 301, 50)),
            fontsize=16,
        )
        ax[i].axhline(
            y=54, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax[i].axhline(
            y=70, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax[i].axhline(
            y=180, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax[i].axhline(
            y=250, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
        )
        ax[i].text(x=1, y=45, s="Severe Hypoglycemia", fontsize=10)
        ax[i].text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=10)
        ax[i].text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=10)
        ax[i].text(x=1, y=255, s="Severe Hyperglycemia", fontsize=10)

        ax[i].legend(prop={"size": 16})

    output_file = graphs_dir / "bgc-abnormal-q-learning-train.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_matplotlib_params()
    graphs_dir = Path.cwd() / "analysis"
    graphs_dir.mkdir(exist_ok=True)
    plot_aggregate(graphs_dir, "G", PopulationType.TRAIN)
    plot_aggregate(graphs_dir, "u", PopulationType.TRAIN)
    plot_aggregate(graphs_dir, "G", PopulationType.TEST)
    plot_aggregate(graphs_dir, "u", PopulationType.TEST)
    plot_abnormal_glycemia(graphs_dir)


if __name__ == "__main__":
    main()
