import math

import matplotlib.pyplot as plt
import numpy as np


def S_f_i(S_f_0: float, t: float) -> float:
    RND = 0.3
    T_s = 5.0
    s = S_f_0 * (
        1
        + 0.3
        * math.sin((2 * math.pi * t) / (24 * (60 / T_s)) + 2 * math.pi * RND)
    )
    return s


def plot_S_f_i(S_f10: float, S_f20: float, S_f30: float) -> None:
    RND = 0.3
    T_s = 5
    t = np.arange(0, 24 * 60, 0.01)

    S_f1_t = S_f10 * (
        1 + np.sin((2 * np.pi * t) / (24 * (60 / T_s)) + 2 * np.pi * RND)
    )
    S_f2_t = S_f20 * (
        1 + np.sin((2 * np.pi * t) / (24 * (60 / T_s)) + 2 * np.pi * RND)
    )

    S_f3_t = S_f30 * (
        1 + np.sin((2 * np.pi * t) / (24 * (60 / T_s)) + 2 * np.pi * RND)
    )

    _, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 12))

    ax[0].plot(t, S_f1_t, color="#36A2EB", linewidth=2, label="S_f1_t")
    ax[1].plot(t, S_f2_t, color="#36A2EB", linewidth=2, label="S_f2_t")
    ax[2].plot(t, S_f3_t, color="#36A2EB", linewidth=2, label="S_f3_t")

    ax[0].axhline(
        y=S_f10, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
    )
    ax[1].axhline(
        y=S_f20, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
    )
    ax[2].axhline(
        y=S_f30, color="#FF6384", alpha=0.5, linewidth=1, linestyle="--"
    )

    plt.show()
