import numpy as np

import patient
from parameters import dt_ctr


def control(
    G_s: float,
    G_s_p: float,
    w: np.ndarray,
    d_hat: np.ndarray,
    fbl: np.ndarray,
    idx: int,
    t: float,
    s: np.ndarray,
    *,
    params: dict,
) -> float:
    b_hat = 1.0
    f_hat = 0
    G_d_p = 0
    G_d_pp = 0
    G_d = params["G_d"] * patient.mg_dL_to_mmol_L

    d_hat[idx] = rbf(s, w, idx, t, params=params)
    fbl[idx] = -(1 / b_hat) * (
        -f_hat
        + G_d_pp
        - 2 * params["lambda"] * (G_s_p - G_d_p)
        - (params["lambda"] ** 2) * (G_s - G_d)
    )

    u = fbl[idx] - (1 / b_hat) * (-d_hat[idx])
    u = max(0, u)

    return u


def rbf(
    s: np.ndarray, w: np.ndarray, idx: int, t: float, *, params: dict
) -> float:
    s_ = s[idx]

    phi = np.exp(
        -(1.0 / 2.0) * np.square((s_ - params["centers"]) / params["widths"])
    )

    if callable(params["eta"]):
        eta = params["eta"](t)
    elif isinstance(params["eta"], float):
        eta = params["eta"]

    w_ = w[idx].copy()
    wp = eta * s_ * phi
    w_norm = np.linalg.norm(w_)
    proj_val = eta * s_ * w_ @ phi
    if not (
        (w_norm < params["w_max"])
        or ((w_norm == params["w_max"]) and (proj_val <= 0))
    ):
        w_matrix = (w_[:, np.newaxis] @ w_[np.newaxis, :]) / (w_ @ w_)
        wp = (np.eye(w_.shape[0]) - w_matrix) @ wp

    w_ += wp * dt_ctr
    d = w_ @ phi

    if idx + 1 < w.shape[0]:
        w[idx + 1] = w_

    return d
