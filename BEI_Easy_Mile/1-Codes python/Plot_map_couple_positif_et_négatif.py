import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Donnees (T=couple, S=vitesse, Z=cos(phi))
# =============================================================================
T_data = np.array([16.583, 18.988, 29.45, 7.644, 32.495, 38.412, 33.2, 15.582, 40.85, 49.164,
                   13.289, 12.272, 60.795, 53.025, 117.075, 47.7, 28.035, 2.5, 10.735, 61.824,
                   23.381, 6.666, 36.181, 57.018, 10.094, 19.488, 48.609, 22.698, 7.505, 31.6,
                   17.85, 28.987, 53.346, 53.53, 52.38, 5.05, 9.595, 20.37, 23.808, 27.371,
                   69.319, 14.847, 24.096, 36.865, 72, 24.735, 31.824, 40.788, 44.352, 46.272,
                   62.304, 76.1, 76.538, 87.87, 83.125, 99.716, 28.684, 87.138, 13.802,
                   37.595, 42.33, 42.874, 52, 63.08, 67.32, 81.279, 86.632, 87.87, 95.76,
                   13.39, 20.055, 22.374, 23.816, 45.374, 44.175, 51.1, 62.296, 73.632,
                   83.712, 18.236, 31.556, 45.186, 46.53, 58.092, 65.52, 77, 93.66, 98.098,
                   29.498, 32.11, 38.178, 54.136, 61.74, 68.02, 16.463, 48.609, 39.382])

S_data = np.array([2852.85, 1468.7, 1230, 2347.2, 956.87, 1260.72, 1607.55, 1890.05, 894.34,
                   596.55, 2195.96, 2371.65, 1432.6, 958.65, 1183.35, 1467.61, 1919.4, 4425.2,
                   2630.4, 938.08, 2238.6, 4092.43, 1821, 1187.76, 2979.2, 2493.63, 1795,
                   4122.82, 3636, 2126.05, 2834, 2802.8, 2609.84, 2687.36, 1705.25, 4382.65,
                   4425.2, 4066.02, 3392.64, 3498.66, 2209.66, 4309.52, 2710, 2126.05,
                   1183.05, 2882.88, 3473.91, 3079.65, 2269.44, 2558.78, 1783, 887, 1426.56,
                   1403.15, 1709.73, 1116.47, 2530.5, 1451.38, 3573.9, 2469.94, 2470,
                   2754.05, 2163.84, 1942.75, 1564.16, 1233.44, 1819.65, 550.2, 1517.19,
                   4387.76, 3763.2, 3991.55, 3063.06, 2096, 2922.3, 1975.05, 1842.67,
                   1926.6, 875.67, 4058.48, 2573.76, 2744.56, 2849, 2020.76, 2209.66,
                   2075.84, 1225.12, 1392.96, 3074.55, 2907.66, 2586.99, 2271.74, 2309,
                   1757, 3983.04, 2376.53, 2546.88])

Z_data = np.array([0.2037, 0.4264, 0.5562, 0.5916, 0.5723, 0.5917, 0.6014, 0.6534, 0.627,
                   0.6767, 0.735, 0.6745, 0.7056, 0.7154, 0.7008, 0.7178, 0.7828, 0.8085,
                   0.78, 0.8, 0.7872, 0.83, 0.83, 0.8051, 0.798, 0.8484, 0.8148, 0.867,
                   0.8772, 0.8858, 0.8787, 0.87, 0.9135, 0.8439, 0.8526, 0.836, 0.88, 0.88,
                   0.8976, 0.8976, 0.88, 0.8633, 0.8633, 0.9167, 0.8544, 0.855, 0.927, 0.855,
                   0.9, 0.927, 0.873, 0.918, 0.918, 0.927, 0.945, 0.945, 0.9464, 0.9373,
                   0.9476, 0.966, 0.966, 0.9292, 0.8832, 0.9108, 0.966, 0.9384, 0.8832,
                   0.9384, 0.92, 0.9021, 0.9021, 0.93, 0.9021, 0.9579, 0.9486, 0.9579,
                   0.9207, 0.9765, 0.8835, 0.893, 0.987, 0.9494, 0.893, 0.94, 0.9588, 0.987,
                   0.9024, 0.893, 0.912, 0.912, 0.9792, 0.96, 0.9312, 0.96, 0.9409, 0.9409,
                   0.9595])

# =============================================================================
# Modele "full" (comme ton code):
# Z = a*T^2 + b*S^2 + c*T*S + d*T + e*S + f
# =============================================================================
def build_A_full(T, S):
    T = np.asarray(T).ravel()
    S = np.asarray(S).ravel()
    return np.column_stack([T**2, S**2, T*S, T, S, np.ones_like(T)])

def fit_full(T, S, Z):
    A = build_A_full(T, S)
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    Z_hat = A @ coeffs
    err = Z_hat - Z

    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    denom = float(np.sum((Z - np.mean(Z))**2))
    r2 = float(1.0 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")

    return coeffs, Z_hat, err, rmse, mae, r2

def predict_full(coeffs, T, S):
    T = np.asarray(T)
    S = np.asarray(S)
    if T.shape != S.shape:
        raise ValueError("T et S doivent avoir la meme forme.")
    A = build_A_full(T, S)
    return (A @ coeffs).reshape(T.shape)

def print_fit(name, coeffs, rmse, mae, r2):
    labels = ["a(T^2)", "b(S^2)", "c(TS)", "d(T)", "e(S)", "f(1)"]
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    for lab, val in zip(labels, coeffs):
        print(f"{lab:<8s} = {val: .6e}")
    print("-" * 70)
    print(f"RMSE = {rmse:.6f} | MAE = {mae:.6f} | R2 = {r2:.6f}")
    print("=" * 70)

# =============================================================================
# Fonction demandee: 2 ajustements + collage + trace 3D final
# =============================================================================
def fit_two_sides_and_plot(T_pos, S_pos, Z_pos, grid_n=120, clip_01=False):
    """
    1) Fit sur T>0 (donnees existantes)
    2) Fit sur T<0 (donnees miroir: -T_pos)
    3) Fonction finale piecewise:
        f(T,S) = f_pos(T,S) si T>=0
               = f_neg(T,S) si T<0
    4) Tracer la surface finale sur [-Tmax, +Tmax]
    """
    # --- Ajustement cote positif
    coeff_pos, Zhat_pos, err_pos, rmse_pos, mae_pos, r2_pos = fit_full(T_pos, S_pos, Z_pos)

    # --- Ajustement cote negatif (miroir)
    T_neg = -T_pos
    coeff_neg, Zhat_neg, err_neg, rmse_neg, mae_neg, r2_neg = fit_full(T_neg, S_pos, Z_pos)

    # --- Affichage resultats
    print_fit("AJUSTEMENT #1 (Couples positifs)", coeff_pos, rmse_pos, mae_pos, r2_pos)
    print_fit("AJUSTEMENT #2 (Couples negatifs - donnees miroir)", coeff_neg, rmse_neg, mae_neg, r2_neg)

    # --- Fonction finale "collee"
    def f_final(T, S):
        T = np.asarray(T)
        S = np.asarray(S)
        Zp = predict_full(coeff_pos, T, S)
        Zn = predict_full(coeff_neg, T, S)
        return np.where(T >= 0, Zp, Zn)

    # --- Tracé 3D de la surface finale
    Tmax = float(np.max(np.abs(T_pos)))
    t_grid = np.linspace(-Tmax, Tmax, grid_n)
    s_grid = np.linspace(float(np.min(S_pos)), float(np.max(S_pos)), grid_n)
    Tm, Sm = np.meshgrid(t_grid, s_grid)

    Zm = f_final(Tm, Sm)
    if clip_01:
        Zm = np.clip(Zm, 0.0, 1.0)

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Tm, Sm, Zm, cmap="viridis", alpha=0.62, edgecolor="none")
    ax.plot_wireframe(Tm, Sm, Zm, rstride=10, cstride=10, linewidth=0.35, alpha=0.25)

    # Points: mesures positives + points miroirs
    ax.scatter(T_pos,  S_pos, Z_pos, s=18, color="red",    label="Mesures (T>0)")
    ax.scatter(-T_pos, S_pos, Z_pos, s=18, color="orange", label="Miroir (T<0)")

    ax.set_xlabel("Couple (Nm)")
    ax.set_ylabel("Vitesse (rpm)")
    ax.set_zlabel("Cos(phi)")
    ax.set_title("Surface finale = collage (fit positif / fit negatif)")

    ax.view_init(elev=22, azim=-55)
    try:
        ax.set_box_aspect((1.2, 1.7, 0.7))
    except Exception:
        pass

    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.06)
    cbar.set_label("Cos(phi)")

    ax.legend(loc="upper right")
    plt.show()

    return coeff_pos, coeff_neg, f_final

# =============================================================================
# Execution
# =============================================================================
if __name__ == "__main__":
    coeff_pos, coeff_neg, f_final = fit_two_sides_and_plot(T_data, S_data, Z_data, grid_n=130, clip_01=False)
