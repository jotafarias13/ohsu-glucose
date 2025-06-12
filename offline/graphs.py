from pathlib import Path

import matplotlib.pyplot as plt

from parameters import DAYS

right = DAYS * 24


def setup_matplotlib_params() -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"


def plot_glucoses(data: dict, output_file: Path) -> None:
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

    ax.text(x=1, y=45, s="Severe Hypoglycemia", fontsize=12)
    ax.text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=12)
    ax.text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=12)
    ax.text(x=1, y=255, s="Severe Hyperglycemia", fontsize=12)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_bgc(data: dict, output_file: Path) -> None:
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

    ax.text(x=1, y=45, s="Severe Hypoglycemia", fontsize=12)
    ax.text(x=1, y=59, s="Moderate Hypoglycemia", fontsize=12)
    ax.text(x=1, y=185, s="Moderate Hyperglycemia", fontsize=12)
    ax.text(x=1, y=255, s="Severe Hyperglycemia", fontsize=12)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_cho(data: dict, output_file: Path) -> None:
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

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_insulin(data: dict, output_file: Path) -> None:
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

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_control(data: dict, output_file: Path) -> None:
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

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_phase_plan(data: dict, output_file: Path) -> None:
    data_ = data

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    scatter = ax.scatter(
        data_["G"], data_["G_p"], c=data_["u"], cmap="viridis", s=5
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Insulin")

    ax.set_xlabel("BGC", labelpad=20, fontsize=16)
    ax.set_ylabel("BGCp", labelpad=20, fontsize=16)

    ax.grid(True)
    fig.tight_layout()

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_error(data: dict, output_file: Path) -> None:
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


def plot_combined_error(data: dict, output_file: Path) -> None:
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

    # interstitial and blood glucose
    graph_file = "interstitial-blood-glucose.pdf"
    plot_glucoses(data, graphs_dir / graph_file)

    # blood glucose
    graph_file = "blood-glucose.pdf"
    plot_bgc(data, graphs_dir / graph_file)

    # gut absorption
    graph_file = "gut-absorption.pdf"
    plot_cho(data, graphs_dir / graph_file)

    # insulin
    graph_file = "plasma-insulin.pdf"
    plot_insulin(data, graphs_dir / graph_file)

    # control
    graph_file = "control-input.pdf"
    plot_control(data, graphs_dir / graph_file)

    # space state
    graph_file = "phase-plan.pdf"
    plot_phase_plan(data, graphs_dir / graph_file)

    # s - combined error
    graph_file = "error.pdf"
    plot_error(data, graphs_dir / graph_file)

    # s - combined error
    graph_file = "combined-error.pdf"
    plot_combined_error(data, graphs_dir / graph_file)

    # glucoses derivatives
    graph_file = "glucoses-derivatives.pdf"
    plot_glucoses_der(data, graphs_dir / graph_file)
