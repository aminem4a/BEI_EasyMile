from __future__ import annotations

import numpy as np
import pandas as pd


def load_map_data(csv_path: str):
    df = pd.read_csv(csv_path)
    T = df["T"].to_numpy(float)
    S = df["S"].to_numpy(float)
    Z = df["Z"].to_numpy(float)
    return T, S, Z


def build_A_full(T, S):
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T * S, T, S, np.ones_like(T)])


def fit_full(T, S, Z):
    A = build_A_full(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    Z_hat = A @ coeffs
    err = Z_hat - Z

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err**2)))  # conservé identique à ton script
    denom = float(np.sum((Z - np.mean(Z)) ** 2))
    r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")
    return coeffs, rmse, mae, r2


def predict_full(coeffs, T, S):
    T = np.asarray(T)
    S = np.asarray(S)
    if T.shape != S.shape:
        raise ValueError("T et S doivent avoir la meme forme.")
    A = build_A_full(T, S)
    return (A @ coeffs).reshape(T.shape)


def build_mapping_two_sides(T_pos, S_pos, Z_pos, clip_01=False, verbose=True):
    coeff_pos, rmse_pos, mae_pos, r2_pos = fit_full(T_pos, S_pos, Z_pos)
    coeff_neg, rmse_neg, mae_neg, r2_neg = fit_full(-T_pos, S_pos, Z_pos)

    if verbose:
        print("\n" + "=" * 80)
        print(f"Fit + : RMSE={rmse_pos:.6f} | MAE={mae_pos:.6f} | R2={r2_pos:.6f}")
        print(f"Fit - : RMSE={rmse_neg:.6f} | MAE={mae_neg:.6f} | R2={r2_neg:.6f}")
        print("=" * 80 + "\n")

    def f_final(T, S):
        T = np.asarray(T)
        S = np.asarray(S)
        Zp = predict_full(coeff_pos, T, S)
        Zn = predict_full(coeff_neg, T, S)
        Z = np.where(T >= 0, Zp, Zn)
        if clip_01:
            Z = np.clip(Z, 0.0, 1.0)
        return Z

    return f_final
