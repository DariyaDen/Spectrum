import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# INPUT DATA (your measurements)
# -----------------------------
PZ = np.array([1750, 1800, 1900, 2000], dtype=float)
FWHM = np.array([8.82, 8.81, 8.57, 8.34], dtype=float)

# -----------------------------
# LINEAR FIT
# -----------------------------
coeffs = np.polyfit(PZ, FWHM, 1)
slope, intercept = coeffs

fit_line = slope * PZ + intercept

# -----------------------------
# METRICS
# -----------------------------
reduction_total = FWHM[0] - FWHM[-1]
reduction_percent = (reduction_total / FWHM[0]) * 100

# -----------------------------
# TIMESTAMP
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -----------------------------
# SAVE CSV
# -----------------------------
csv_filename = f"fwhm_vs_pz_{timestamp}.csv"
with open(csv_filename, "w") as f:
    f.write("PZ,FWHM\n")
    for p, fwhm in zip(PZ, FWHM):
        f.write(f"{p},{fwhm}\n")

# -----------------------------
# SAVE TEXT REPORT
# -----------------------------
report_filename = f"fit_report_{timestamp}.txt"
with open(report_filename, "w") as f:
    f.write("=== FWHM vs PZ Analysis ===\n\n")

    f.write("Data points:\n")
    for p, fw in zip(PZ, FWHM):
        f.write(f"PZ = {p}, FWHM = {fw:.4f} keV\n")

    f.write("\n--- Linear fit ---\n")
    f.write(f"FWHM(PZ) = {intercept:.4f} + ({slope:.6f}) * PZ\n")

    f.write("\n--- Reduction ---\n")
    f.write(f"Total reduction: {reduction_total:.4f} keV\n")
    f.write(f"Relative reduction: {reduction_percent:.2f} %\n")

    f.write("\n--- Interpretation ---\n")
    f.write("Monotonic decrease indicates incomplete pole-zero optimization.\n")
    f.write("Weak slope suggests other effects (ballistic deficit) dominate.\n")

# -----------------------------
# PLOT
# -----------------------------
plt.figure()

plt.scatter(PZ, FWHM, label="Data")
plt.plot(PZ, fit_line, linestyle="--", label="Linear fit")

plt.xlabel("Pole-zero (PZ)")
plt.ylabel("FWHM (keV)")
plt.title("FWHM vs Pole-zero")

plt.legend()
plt.grid()

plot_filename = f"fwhm_plot_{timestamp}.png"
plt.savefig(plot_filename, dpi=300)

print("\n=== DONE ===")
print(f"Saved CSV: {csv_filename}")
print(f"Saved report: {report_filename}")
print(f"Saved plot: {plot_filename}")
