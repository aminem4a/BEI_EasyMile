import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# ============================================================
# 1) LECTURE DES DONNÉES SIMPLIFIÉES 
# ============================================================

data_ref_simpli = pd.read_excel('modeli_simplifie.xlsx',sheet_name='reff',header=None).values

data_feed_simpli = pd.read_excel('modeli_simplifie.xlsx',sheet_name='feedB',header=None).values

# ============================================================
# 2 EXTRACTION DES SIGNAUX SIMPLIFIÉS 
# ============================================================
time_ref = data_ref_simpli[:, 6] - data_ref_simpli[0, 6]

C_ref = data_ref_simpli[:, 7]

time_feed = data_feed_simpli[:, 1] - data_feed_simpli[0, 1]

C_feed = data_feed_simpli[:, 4]

# ============================================================
# 3 LECTURE DES DONNÉES COMPLÈTES 
# ============================================================

data_ref_full = pd.read_excel(
    'Motor_StepResponse-1.xlsx',
    sheet_name='Axle_torque_setpoint',
    header=0           
).values

data_feed_full = pd.read_excel(
    'Motor_StepResponse-1.xlsx',
    sheet_name='Axle_torque_feedback',
    header=0           
).values

# ============================================================
# 4) EXTRACTION DES SIGNAUX COMPLETS
# ============================================================

time_ref_full = data_ref_full[:, 6] - data_ref_simpli[0, 6]

C_ref_full = data_ref_full[:, 7]

time_feed_full = data_feed_full[:, 1] - data_feed_simpli[0, 1]

C_feed_full = data_feed_full[:, 4]

C_ref_full_d = np.column_stack((time_ref_full, C_ref_full))

C_feed_full_d = np.column_stack((time_feed_full, C_feed_full))

# ============================================================
# 5) TRACÉ TYPE "SCOPE" – DONNÉES COMPLÈTES
# ============================================================

plt.figure()
plt.plot(C_ref_full_d[:, 0], C_ref_full_d[:, 1],
         label='Couple de référence', linewidth=2)
plt.plot(C_feed_full_d[:, 0], C_feed_full_d[:, 1],
         label='Couple mesuré', linewidth=2)
plt.xlabel('Temps (s)')
plt.ylabel('Couple')
plt.title('Réponse en couple – Référence vs Feedback')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 6) TRACÉ DES DONNÉES SIMPLIFIÉES
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(time_ref, C_ref, 'b', linewidth=2, label='C_ref (setpoint)')
plt.plot(time_feed, C_feed, 'r', linewidth=2, label='C_feed (feedback)', alpha=0.7)
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Setpoint et Feedback - Données simplifiées')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 7) DÉTECTION DU FRONT DE STEP
# ============================================================

dy = np.diff(C_ref)
idx_step = np.argmax(np.abs(dy))

if idx_step + 1 < len(time_ref):
    t_step = time_ref[idx_step + 1]
else:
    t_step = time_ref[-1]

print(f"Step détecté à l'index {idx_step}, temps = {t_step:.6f} s")
print(f"C_ref[{idx_step}] = {C_ref[idx_step]:.2f}")
print(f"C_ref[{idx_step+1}] = {C_ref[idx_step+1]:.2f}")

# ============================================================
# 8) FENÊTRAGE AUTOUR DU STEP
# ============================================================

t_before = 0.2
t_after = 2.0

f_interp = interpolate.interp1d(
    time_ref, C_ref,
    kind='linear',
    fill_value='extrapolate',
    bounds_error=False
)
C_ref_interp = f_interp(time_feed)

mask = (time_feed >= t_step - t_before) & (time_feed <= t_step + t_after)

if np.sum(mask) < 2:
    mask = (time_feed >= t_step)

start_idx = np.where(mask)[0][0]

t = time_feed[mask] - time_feed[start_idx]
yf = C_feed[mask]
yr = C_ref_interp[mask]

# ============================================================
# 9) MÉTHODE DE STREJC
# ============================================================

k = yf[-1] / yr[-1]
dt = t[1] - t[0]

dyf = np.diff(yf) / dt
dyf = np.append(dyf, 0)

dyfmax_idx = np.argmax(dyf)
dyfmax = dyf[dyfmax_idx]
tmax = t[dyfmax_idx]
ypi = yf[dyfmax_idx]

a = dyfmax
b = ypi - a * tmax

t1 = -b / a if a != 0 else 0
t2 = (k - b) / a if a != 0 else 0

Td = t1
Ta = t2 - t1

non_zero_idx = np.where(yf != 0)[0]
Tr = t[non_zero_idx[0]] if len(non_zero_idx) > 0 else 0

Tu = Td - Tr
rapport = Tu / Ta if Ta != 0 else 0
n = 2

tau = Ta / np.exp(1)
Tutab = Ta * 0.104
TR = Tu - Tutab
T = max(Tr + TR, 0)

# ============================================================
# 10) AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n" + "="*60)
print("RÉSULTATS DU MODÈLE STREJC")
print("="*60)
print(f"k     = {k:.4f}")
print(f"Td    = {Td:.4f} s")
print(f"Ta    = {Ta:.4f} s")
print(f"n     = {n}")
print(f"tau   = {tau:.4f} s")
print(f"T     = {T:.4f} s")
print(f"Tu    = {Tu:.4f} s")
print(f"Tr    = {Tr:.4f} s")
print(f"rapport = {rapport:.4f}")
print(f"tau1     = 30")