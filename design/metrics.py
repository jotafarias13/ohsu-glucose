from pathlib import Path

import numpy as np


def exclude_training(data: dict, params: dict) -> dict:
    idx = np.searchsorted(data["time"], 24 * params["days_training"])

    return {k: v[idx:] for k, v in data.items()}


def calculate_metrics(data: dict) -> dict:
    bgc_mean = data["G"].mean()
    bgc_max = data["G"].max()
    bgc_min = data["G"].min()
    bgc_sd = data["G"].std()
    bgc_cv = bgc_sd / bgc_mean

    dt = data["time"][1] - data["time"][0]
    total_hours = data["time"][-1] - data["time"][0] + dt
    days = total_hours / 24
    insulin_day = np.trapezoid(data["u"], data["time"]) / days

    hiper_sev, hiper_mod = 250, 180
    hipo_mod, hipo_sev = 70, 54
    tar_sev = (data["G"] > hiper_sev).sum()
    tar_mod = ((data["G"] > hiper_mod) & (data["G"] < hiper_sev)).sum()
    tbr_sev = (data["G"] < hipo_sev).sum()
    tbr_mod = ((data["G"] < hipo_mod) & (data["G"] > hipo_sev)).sum()
    tir = ((data["G"] > hipo_mod) & (data["G"] < hiper_mod)).sum()

    tar_sev /= data["time"].shape[0]
    tar_mod /= data["time"].shape[0]
    tir /= data["time"].shape[0]
    tbr_mod /= data["time"].shape[0]
    tbr_sev /= data["time"].shape[0]

    metrics = {
        "bgc_mean": float(bgc_mean),
        "bgc_max": float(bgc_max),
        "bgc_min": float(bgc_min),
        "bgc_sd": float(bgc_sd),
        "bgc_cv": float(bgc_cv),
        "insulin_day": float(insulin_day),
        "tar_sev": float(tar_sev),
        "tar_mod": float(tar_mod),
        "tir": float(tir),
        "tbr_mod": float(tbr_mod),
        "tbr_sev": float(tbr_sev),
    }

    return metrics


def pprint_metrics(metrics: dict) -> None:
    text = f"""\
    Glycemia
    \tMean: {metrics["bgc_mean"]:.2f} mg/dL
    \tSD: {metrics["bgc_sd"]:.2f} mg/dL   (< 50)
    \tCV: {metrics["bgc_cv"]:.2%}   (< 36%)
    \tMax: {metrics["bgc_max"]:.2f} mg/dL
    \tMin: {metrics["bgc_min"]:.2f} mg/dL

    Insulin
    \tMean 24h: {metrics["insulin_day"]:.2f} U

    Time in Range
    \tTAR Severe: {metrics["tar_sev"]:.2%}   (< 5%)
    \tTAR Moderate: {metrics["tar_mod"]:.2%}   (< 25%)
    \tTIR: {metrics["tir"]:.2%}   (> 70%)
    \tTBR Moderate: {metrics["tbr_mod"]:.2%}   (< 4%)
    \tTBR Severe: {metrics["tbr_sev"]:.2%}   (< 1%)\
    """
    text = "#" * 40 + f"\n{text}\n" + "#" * 40
    print(text)


def print_metrics(metrics: dict) -> None:
    text = (
        "{"
        f"bgc_mean: {metrics['bgc_mean']:.2f}, "
        f"bgc_max: {metrics['bgc_max']:.2f}, "
        f"bgc_min: {metrics['bgc_min']:.2f}, bgc_sd: {metrics['bgc_sd']:.2f}, "
        f"bgc_cv: {metrics['bgc_cv']:.2%}, "
        f"insulin_day: {metrics['insulin_day']:.2f}, "
        f"tar_sev: {metrics['tar_sev']:.2%}, "
        f"tar_mod: {metrics['tar_mod']:.2%}, "
        f"tir: {metrics['tir']:.2%}, tbr_mod: {metrics['tbr_mod']:.2%}, "
        f"tbr_sev: {metrics['tbr_sev']:.2%}"
        "}"
    )

    print(text)


def save_metrics(metrics: dict, graphs_dir: Path) -> None:
    text = f"""\
    Glycemia
    \tMean: {metrics["bgc_mean"]:.2f} mg/dL
    \tSD: {metrics["bgc_sd"]:.2f} mg/dL   (< 50)
    \tCV: {metrics["bgc_cv"]:.2%}   (< 36%)
    \tMax: {metrics["bgc_max"]:.2f} mg/dL
    \tMin: {metrics["bgc_min"]:.2f} mg/dL

    Insulin
    \tMean 24h: {metrics["insulin_day"]:.2f} U

    Time in Range
    \tTAR Severe: {metrics["tar_sev"]:.2%}   (< 5%)
    \tTAR Moderate: {metrics["tar_mod"]:.2%}   (< 25%)
    \tTIR: {metrics["tir"]:.2%}   (> 70%)
    \tTBR Moderate: {metrics["tbr_mod"]:.2%}   (< 4%)
    \tTBR Severe: {metrics["tbr_sev"]:.2%}   (< 1%)\
    """
    text = "#" * 40 + f"\n{text}\n" + "#" * 40

    metrics_file = "metrics.txt"
    (graphs_dir / metrics_file).write_text(text)
