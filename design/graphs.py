from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import exclude_training
from parameters import ControllerType
from patient import Patient


def setup_matplotlib_params() -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"


def plot_glucoses(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["time"], data["G"], color="#FF6384", linewidth=2, label="$G(t)$"
    )
    ax.plot(
        data["time"],
        data["G_s"],
        color="#36A2EB",
        linewidth=2,
        label="$G_{s}(t)$",
    )

    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel(
        (
            "Interstitial and Blood Glucose Concentration "
            r"$\left[ \si{mg {\cdot} dL^{-1}} \right]$"
        ),
        labelpad=20,
        fontsize=16,
    )

    ax.set_xlim(left=0, right=right)
    ax.set_ylim(bottom=-100, top=315)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax.set_yticks(
        ticks=list(range(0, 301, 50)),
        labels=list(range(0, 301, 50)),
        fontsize=16,
    )

    ax.axvline(x=24, color="#C9C9C9", linewidth=2)
    ax.axvline(x=48, color="#C9C9C9", linewidth=2)

    ax.axhline(y=54, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--")
    ax.axhline(y=70, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--")
    ax.axhline(y=180, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--")
    ax.axhline(y=250, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--")

    ax.legend(prop={"size": 16})

    ax.text(x=11, y=290, s="Day 1", fontsize=16)
    ax.text(x=35, y=290, s="Day 2", fontsize=16)
    ax.text(x=59, y=290, s="Day 3", fontsize=16)

    ax.text(x=1, y=45, s="Severe Hypoglycemia", fontsize=12)
    ax.text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=12)
    ax.text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=12)
    ax.text(x=1, y=255, s="Severe Hyperglycemia", fontsize=12)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_bgc(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["time"], data["G"], color="#36A2EB", linewidth=4, label="$G(t)$"
    )

    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel(
        (
            "Blood Glucose Concentration "
            r"$\left[ \si{mg {\cdot} dL^{-1}} \right]$"
        ),
        labelpad=20,
        fontsize=16,
    )

    ax.set_xlim(left=0, right=right)
    # ax.set_ylim(bottom=-100, top=315)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax.set_yticks(
        ticks=list(range(0, 301, 50)),
        labels=list(range(0, 301, 50)),
        fontsize=16,
    )

    ax.axvline(x=24, color="#C9C9C9", linewidth=2)
    ax.axvline(x=48, color="#C9C9C9", linewidth=2)

    ax.axhline(y=54, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--")
    ax.axhline(y=70, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--")
    ax.axhline(y=180, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--")
    ax.axhline(y=250, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--")

    ax.legend(prop={"size": 16})

    ax.text(x=11, y=290, s="Day 1", fontsize=16)
    ax.text(x=35, y=290, s="Day 2", fontsize=16)
    ax.text(x=59, y=290, s="Day 3", fontsize=16)

    ax.text(x=1, y=45, s="Severe Hypoglycemia", fontsize=12)
    ax.text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=12)
    ax.text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=12)
    ax.text(x=1, y=255, s="Severe Hyperglycemia", fontsize=12)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_cho(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["time"],
        data["U_G"],
        color="#36A2EB",
        linewidth=4,
        label="$U_{G}(t)$",
    )

    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel(
        (
            "Gut Absorption Rate "
            r"$\left[ \si{mmol {\cdot} kg^{-1} {\cdot} min^{-1}} \right]$"
        ),
        labelpad=20,
        fontsize=16,
    )

    ax.set_xlim(left=0, right=right)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax.tick_params(axis="y", labelsize=16)

    ax.axvline(x=24, color="#C9C9C9", linewidth=2)
    ax.axvline(x=48, color="#C9C9C9", linewidth=2)

    ax.legend(prop={"size": 16})

    ax.text(x=11, y=0.043, s="Day 1", fontsize=16)
    ax.text(x=32, y=0.043, s="Day 2", fontsize=16)
    ax.text(x=59, y=0.043, s="Day 3", fontsize=16)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_insulin(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["time"],
        data["I"],
        color="#36A2EB",
        linewidth=4,
        label="$I(t)$",
    )

    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel(
        (
            "Plasma Insulin Concentration"
            r"$\left[ \si{mU {\cdot} L^{-1}} \right]$"
        ),
        labelpad=20,
        fontsize=16,
    )

    ax.set_xlim(left=0, right=right)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax.tick_params(axis="y", labelsize=16)

    ax.axvline(x=24, color="#C9C9C9", linewidth=2)
    ax.axvline(x=48, color="#C9C9C9", linewidth=2)

    ax.legend(prop={"size": 16})

    ax.text(x=0.15, y=0.95, s="Day 1", fontsize=16, transform=ax.transAxes)
    ax.text(x=0.48, y=0.95, s="Day 2", fontsize=16, transform=ax.transAxes)
    ax.text(x=0.81, y=0.95, s="Day 3", fontsize=16, transform=ax.transAxes)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_control(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["time"],
        data["u"],
        color="#36A2EB",
        linewidth=2,
        label="$u(t)$",
    )

    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel(
        ("Insulin Infusion" r"$\left[ \si{U {\cdot} h^{-1}} \right]$"),
        labelpad=20,
        fontsize=16,
    )

    ax.set_xlim(left=0, right=right)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax.tick_params(axis="y", labelsize=16)

    ax.axvline(x=24, color="#C9C9C9", linewidth=2)
    ax.axvline(x=48, color="#C9C9C9", linewidth=2)

    ax.legend(prop={"size": 16})

    ax.text(x=0.15, y=0.97, s="Day 1", fontsize=16, transform=ax.transAxes)
    ax.text(x=0.48, y=0.97, s="Day 2", fontsize=16, transform=ax.transAxes)
    ax.text(x=0.81, y=0.97, s="Day 3", fontsize=16, transform=ax.transAxes)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_bgc_control(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))

    ax[0].plot(
        data["time"], data["G"], color="#36A2EB", linewidth=4, label="$G(t)$"
    )
    ax[0].set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax[0].set_ylabel(
        (
            "Blood Glucose Concentration "
            r"$\left[ \si{mg {\cdot} dL^{-1}} \right]$"
        ),
        labelpad=20,
        fontsize=16,
    )
    ax[0].set_xlim(left=0, right=right)
    ax[0].set_ylim(0, 400)
    ax[0].set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax[0].set_yticks(
        ticks=list(range(0, 301, 50)),
        labels=list(range(0, 301, 50)),
        fontsize=16,
    )
    ax[0].axhline(
        y=54, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
    )
    ax[0].axhline(
        y=70, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--"
    )
    ax[0].axhline(
        y=180, color="#4BC0C0", alpha=0.5, linewidth=1, linestyle="--"
    )
    ax[0].axhline(
        y=250, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
    )
    ax[0].text(x=1, y=45, s="Severe Hypoglycemia", fontsize=12)
    ax[0].text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=12)
    ax[0].text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=12)
    ax[0].text(x=1, y=255, s="Severe Hyperglycemia", fontsize=12)
    ax[0].legend(prop={"size": 16})

    ax[1].plot(
        data["time"],
        data["u"],
        color="#36A2EB",
        linewidth=4,
        label="$u(t)$",
    )
    ax[1].set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax[1].set_ylabel(
        ("Insulin Infusion" r"$\left[ \si{U {\cdot} h^{-1}} \right]$"),
        labelpad=20,
        fontsize=16,
    )
    ax[1].set_xlim(left=0, right=right)
    ax[1].set_ylim(-0.2, 6.0)
    ax[1].set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax[1].tick_params(axis="y", labelsize=16)
    ax[1].legend(prop={"size": 16})

    for day in range(DAYS):
        ax[0].axvline(x=24 * (day + 1), color="#C9C9C9", linewidth=2)
        ax[0].text(x=10 + 24 * day, y=350, s=f"Day {day + 1}", fontsize=16)
        ax[1].axvline(x=24 * (day + 1), color="#C9C9C9", linewidth=2)
        ax[1].text(x=10 + 24 * day, y=5.2, s=f"Day {day + 1}", fontsize=16)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_exercise(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

    exercise_types = {"M_PGU": (0, 0), "M_PIU": (0, 1), "M_HGP": (1, 0)}

    for value, idx in exercise_types.items():
        val = value.split("_")
        label = rf"${val[0]}_\text{{{val[1]}}}(t)$"
        ax[idx].plot(
            data["time"],
            data[value],
            color="#36A2EB",
            linewidth=1,
            label=label,
        )
        ax[idx].set_xlim(left=0, right=right)
        ax[idx].set_xticks(
            ticks=list(range(0, DAYS * 24 + 1, 6)),
            labels=[0] + [6, 12, 18, 24] * DAYS,
            fontsize=12,
        )
        ax[idx].tick_params(axis="y", labelsize=12)
        ax[idx].axvline(x=24, color="#C9C9C9", linewidth=1)
        ax[idx].axvline(x=48, color="#C9C9C9", linewidth=1)
        ax[idx].legend(
            prop={"size": 10}, loc="upper left", bbox_to_anchor=(0.01, 0.90)
        )
        text_x = [0.03, 0.50, 0.78] if value == "M_PIU" else [0.12, 0.45, 0.78]
        ax[idx].text(
            x=text_x[0],
            y=0.93,
            s="Exercise 1",
            fontsize=12,
            transform=ax[idx].transAxes,
        )
        ax[idx].text(
            x=text_x[1],
            y=0.93,
            s="Exercise 2",
            fontsize=12,
            transform=ax[idx].transAxes,
        )
        ax[idx].text(
            x=text_x[2],
            y=0.93,
            s="Exercise 3",
            fontsize=12,
            transform=ax[idx].transAxes,
        )

    ax[1, 1].plot(
        data["time"], data["G"], color="#36A2EB", linewidth=2, label="$G(t)$"
    )
    ax[1, 1].set_xlim(left=0, right=right)
    ax[1, 1].set_ylim(bottom=-100, top=315)
    ax[1, 1].set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=12,
    )
    ax[1, 1].set_yticks(
        ticks=list(range(0, 301, 50)),
        labels=list(range(0, 301, 50)),
        fontsize=12,
    )
    ax[1, 1].axvline(x=15, color="#C9C9C9", linewidth=1)
    ax[1, 1].axvline(x=34, color="#C9C9C9", linewidth=1)
    ax[1, 1].axvline(x=67, color="#C9C9C9", linewidth=1)
    ax[1, 1].legend(prop={"size": 10}, loc="lower left")

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_weights(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    for weight in range(data["w"].shape[1]):
        ax.plot(
            data["time"],
            data["w"][:, weight],
            linewidth=2,
            label=f"$w_{{{weight + 1}}}(t)$",
        )
    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel("Neural Network Weights", labelpad=20, fontsize=16)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )

    ax.legend(prop={"size": 10}, loc="upper left")

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_weights_norm(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    norms = np.linalg.norm(data["w"], axis=1)
    ax.plot(data["time"], norms, linewidth=2, label="$w(t)$")
    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel("Neural Network Weights Norm", labelpad=20, fontsize=16)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )

    ax.legend(prop={"size": 10})

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_rbf(params: dict, output_file: Path) -> None:
    start = params["centers"][0] - 3 * params["widths"][0]
    end = params["centers"][-1] + 3 * params["widths"][-1]
    length = (end - start) / 10_000
    error = np.arange(start, end, length, dtype=float)
    phi = [
        np.exp(
            -(1.0 / 2.0)
            * np.square((s - params["centers"]) / params["widths"])
        )
        for s in error
    ]
    phi = np.array(phi)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    for phi_ in phi.T:
        ax.plot(error, phi_, linewidth=2)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_estimation(
    data: dict, output_file: Path, patient: Patient, params: dict
) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 15))

    ax[0].plot(
        data["time"], data["d"], color="#FF6384", linewidth=1, label="$d(t)$"
    )
    ax[1].plot(
        data["time"],
        data["d_hat"],
        color="#36A2EB",
        linewidth=1,
        label="$d_{hat}(t)$",
    )
    ax[2].plot(
        data["time"], data["d"], color="#FF6384", linewidth=2, label="$d(t)$"
    )
    ax[2].plot(
        data["time"],
        data["d_hat"],
        color="#36A2EB",
        linewidth=1,
        label="$d_{hat}(t)$",
    )
    ax[3].plot(
        data["time"],
        data["fbl"],
        color="#FF6384",
        linewidth=1,
        label="$fbl(t)$",
    )
    ax[3].plot(
        data["time"],
        data["d_hat"],
        color="#36A2EB",
        linewidth=1,
        label="$d_{hat}(t)$",
    )
    ax[3].plot(
        data["time"],
        data["u"] / patient.mU_kg_min_to_U_h,
        color="#00FF00",
        linewidth=1,
        label="$u(t)$",
    )

    for ax_ in ax:
        ax_.set_ylabel("Estimation", labelpad=20, fontsize=16)
        ax_.set_xlim(left=0, right=right)
        ax_.set_xticks(
            ticks=list(range(0, DAYS * 24 + 1, 6)),
            labels=[0] + [6, 12, 18, 24] * DAYS,
            fontsize=16,
        )
        ax_.tick_params(axis="y", labelsize=16)
        ax_.axvline(x=24, color="#C9C9C9", linewidth=2)
        ax_.axvline(x=48, color="#C9C9C9", linewidth=2)
        ax_.legend(prop={"size": 10}, loc="upper left")
        ax_.text(
            x=0.15, y=0.90, s="Day 1", fontsize=16, transform=ax_.transAxes
        )
        ax_.text(
            x=0.48, y=0.90, s="Day 2", fontsize=16, transform=ax_.transAxes
        )
        ax_.text(
            x=0.81, y=0.90, s="Day 3", fontsize=16, transform=ax_.transAxes
        )

    ax[2].set_xlabel("Time [h]", labelpad=20, fontsize=16)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_phase_plan(data: dict, output_file: Path, params: dict) -> None:
    data_ = exclude_training(data, params)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    scatter = ax.scatter(
        data_["G"],
        data_["G_p"],
        c=data_["u"],
        cmap="inferno",
        s=5,
        vmin=0.0,
        vmax=3.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Insulin")

    ax.set_xlabel("BGC", labelpad=20, fontsize=16)
    ax.set_ylabel("BGCp", labelpad=20, fontsize=16)

    ax.grid(True)
    fig.tight_layout()

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_error(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["time"],
        data["s"],
        color="#36A2EB",
        linewidth=2,
        label="$s(t)$",
    )

    ax.set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax.set_ylabel("Error", labelpad=20, fontsize=16)

    ax.set_xlim(left=0, right=right)

    ax.set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(prop={"size": 16})

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_combined_error(data: dict, output_file: Path, params: dict) -> None:
    DAYS = params["days"]
    right = DAYS * 24

    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

    ax[0].plot(
        data["time"],
        data["s"],
        color="#36A2EB",
        linewidth=2,
        label="$s(t)$",
    )

    ax[0].set_xlabel("Time [h]", labelpad=20, fontsize=16)
    ax[0].set_ylabel("Error", labelpad=20, fontsize=16)

    ax[0].set_xlim(left=0, right=right)

    ax[0].set_xticks(
        ticks=list(range(0, DAYS * 24 + 1, 6)),
        labels=[0] + [6, 12, 18, 24] * DAYS,
        fontsize=16,
    )
    ax[0].tick_params(axis="y", labelsize=16)

    ax[1].plot(
        data["time"],
        data["G_p_error"],
        color="#36A2EB",
        linewidth=2,
        label="G_p_error",
    )
    ax[1].plot(
        data["time"],
        data["l_G_error"],
        color="#FF6384",
        linewidth=1,
        label="l_G_error",
    )

    ax[0].legend(prop={"size": 16})
    ax[1].legend(prop={"size": 16})

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_estimation_s(data: dict, output_file: Path) -> None:
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.plot(
        data["s"],
        data["d_hat"],
        color="#36A2EB",
        linewidth=2,
        label="$d_{hat}(t)$",
    )

    ax.set_xlabel("Error [s]", labelpad=20, fontsize=16)
    ax.set_ylabel("Estimation", labelpad=20, fontsize=16)
    ax.legend(prop={"size": 16})
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_glucoses_der(data: dict, output_file: Path) -> None:
    _, ax = plt.subplots(3, 1, figsize=(16, 12))
    ax[0].plot(data["time"], data["G"], color="#36A2EB", label="G(t)")
    ax[1].plot(data["time"], data["G_p"], color="#FF6384", label="Gp(t)")
    ax[2].plot(data["time"], data["G_s_pp"], color="#00FF00", label="Gpp(t)")

    ax[0].legend(prop={"size": 16})
    ax[1].legend(prop={"size": 16})
    ax[2].legend(prop={"size": 16})
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def run(data: dict, params: dict) -> None:
    setup_matplotlib_params()
    graphs_dir = params["graphs_dir"]
    graphs_dir.mkdir(exist_ok=True)

    patient = Patient()

    # interstitial and blood glucose
    graph_file = "interstitial-blood-glucose.pdf"
    plot_glucoses(data, graphs_dir / graph_file, params)

    # blood glucose
    graph_file = "blood-glucose.pdf"
    plot_bgc(data, graphs_dir / graph_file, params)

    # gut absorption
    graph_file = "gut-absorption.pdf"
    plot_cho(data, graphs_dir / graph_file, params)

    # insulin
    graph_file = "plasma-insulin.pdf"
    plot_insulin(data, graphs_dir / graph_file, params)

    # control
    graph_file = "control-input.pdf"
    plot_control(data, graphs_dir / graph_file, params)

    # blood glucose and control
    graph_file = "blood-glucose-and-control.pdf"
    plot_bgc_control(data, graphs_dir / graph_file, params)

    # exercise
    graph_file = "exercise.pdf"
    plot_exercise(data, graphs_dir / graph_file, params)

    if params["controller_type"] != ControllerType.FBL:
        # neural network
        graph_file = "weights.pdf"
        plot_weights(data, graphs_dir / graph_file, params)

        # neural network weights norm
        graph_file = "weights-norm.pdf"
        plot_weights_norm(data, graphs_dir / graph_file, params)

        # rbf
        graph_file = "rbf.pdf"
        plot_rbf(params, graphs_dir / graph_file)

    # estimation
    graph_file = "estimation.pdf"
    plot_estimation(data, graphs_dir / graph_file, patient, params)

    # space state
    graph_file = "phase-plan.pdf"
    plot_phase_plan(data, graphs_dir / graph_file, params)

    # s - combined error
    graph_file = "error.pdf"
    plot_error(data, graphs_dir / graph_file, params)

    # s - combined error
    graph_file = "combined-error.pdf"
    plot_combined_error(data, graphs_dir / graph_file, params)

    # estimation s
    graph_file = "estimation-s.pdf"
    plot_estimation_s(data, graphs_dir / graph_file)

    # glucoses derivatives
    graph_file = "glucoses-derivatives.pdf"
    plot_glucoses_der(data, graphs_dir / graph_file)
