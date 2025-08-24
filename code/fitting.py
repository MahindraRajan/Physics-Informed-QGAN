# Fit Fano lineshape (A0, q, w0_THz, Gamma_THz) to 10 random absorption spectra.
# ω is FREQUENCY (THz), not angular frequency.
# Output: fano_fit_results_10_simplified.csv with columns:
# structure_name, A0, q, w0_THz, Gamma_THz

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "absorptionData_HybridGAN.csv"   # path to your CSV
STRUCTURE_COL = "Var1_1"                    # structure name column
WAVELENGTH_MIN_UM = 4.0
WAVELENGTH_MAX_UM = 12.0
N_RANDOM = 10
RANDOM_SEED = 42
OUTPUT_CSV = "fano_fit_results_10_simplified.csv"

# -----------------------------
# Fano model (ω in THz)
# -----------------------------
def fano(omega_THz, A0, q, w0_THz, Gamma_THz):
    """
    Fano lineshape using frequency ω (THz), not angular frequency.
    ε = 2*(ω - ω0)/Γ
    A(ω) = A0 * ((q + ε)^2) / (1 + ε^2)
    """
    eps = 2.0 * (omega_THz - w0_THz) / np.maximum(Gamma_THz, 1e-12)
    return A0 * ((q + eps)**2) / (1.0 + eps**2)

# -----------------------------
# Helpers
# -----------------------------
def build_frequency_axis_THz(n_points, wl_min_um, wl_max_um):
    """
    Your columns Var1_2 ... Var1_(1+n_points) span wl_min_um–wl_max_um (µm), inclusive.
    Convert λ (µm) to ω (THz) via ω = c/λ and return ascending ω with a sorting index.
    """
    lam_um = np.linspace(wl_min_um, wl_max_um, n_points)
    c = 299_792_458.0  # m/s
    omega_THz = (c / (lam_um * 1e-6)) / 1e12
    idx = np.argsort(omega_THz)  # ascending (~25 → 75 THz)
    return omega_THz[idx], idx

def fwhm_estimate(x, y):
    """Rough FWHM estimate for initial Γ."""
    y = np.asarray(y)
    x = np.asarray(x)
    ymax = float(np.nanmax(y))
    if not np.isfinite(ymax) or ymax <= 0:
        return max(1e-3, 0.05*(x.max() - x.min()))
    half = ymax / 2.0
    s = np.sign(y - half)
    crossings = np.where(np.diff(s) != 0)[0]
    if len(crossings) < 2:
        return max(1e-3, 0.05*(x.max() - x.min()))
    def interp(i):
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        if y1 == y0:
            return x0
        t = (half - y0) / (y1 - y0)
        return x0 + t*(x1 - x0)
    xL = interp(crossings[0])
    xR = interp(crossings[-1])
    width = abs(xR - xL)
    if not np.isfinite(width) or width <= 0:
        width = max(1e-3, 0.05*(x.max() - x.min()))
    return width

def initial_guess_and_bounds(omega_THz, A):
    """Heuristic initial guesses and bounds for curve_fit."""
    A = np.asarray(A)
    i_max = int(np.nanargmax(A))
    w0_guess = float(omega_THz[i_max])
    A0_guess = float(np.nanmax(A))
    Gamma_guess = float(fwhm_estimate(omega_THz, A))

    wmin, wmax = float(np.nanmin(omega_THz)), float(np.nanmax(omega_THz))
    span = max(1e-6, wmax - wmin)

    p0 = [A0_guess, 1.0, w0_guess, Gamma_guess]
    bounds_lower = [0.0, -100.0, wmin, 1e-3]
    bounds_upper = [max(10.0, 10*A0_guess if np.isfinite(A0_guess) else 10.0),
                    100.0, wmax, span]
    return p0, (bounds_lower, bounds_upper)

def fit_one(omega_THz, A):
    """Return dict: A0, q, w0_THz, Gamma_THz (NaN if fit fails)."""
    mask = np.isfinite(omega_THz) & np.isfinite(A)
    x = omega_THz[mask]
    y = np.asarray(A)[mask]
    if x.size < 5:
        return dict(A0=np.nan, q=np.nan, w0_THz=np.nan, Gamma_THz=np.nan)
    p0, bounds = initial_guess_and_bounds(x, y)
    try:
        popt, _ = curve_fit(fano, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return dict(A0=float(popt[0]), q=float(popt[1]),
                    w0_THz=float(popt[2]), Gamma_THz=float(popt[3]))
    except Exception:
        return dict(A0=np.nan, q=np.nan, w0_THz=np.nan, Gamma_THz=np.nan)

# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(CSV_PATH)

    # Identify spectral columns Var1_2 ... Var1_801 (by numeric suffix)
    spec_cols = [c for c in df.columns if c.startswith("Var1_") and c != STRUCTURE_COL]
    def _suffix_num(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return None
    spec_cols = [c for c in spec_cols if _suffix_num(c) is not None]
    spec_cols = sorted(spec_cols, key=lambda c: _suffix_num(c))
    n_points = len(spec_cols)
    if n_points < 5:
        raise ValueError(f"Found only {n_points} spectral columns; expected ~800.")

    omega_THz, sort_idx = build_frequency_axis_THz(
        n_points, WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM
    )

    # Choose N_RANDOM unique rows
    rng = np.random.default_rng(RANDOM_SEED)
    chosen_idx = rng.choice(len(df), size=min(N_RANDOM, len(df)), replace=False)

    results = []
    for i in chosen_idx:
        row = df.iloc[i]
        spectrum = row[spec_cols].to_numpy(dtype=float)[sort_idx]
        fit = fit_one(omega_THz, spectrum)
        results.append({
            "structure_name": row.get(STRUCTURE_COL, f"row_{i}"),
            **fit
        })

    out_df = pd.DataFrame(results, columns=["structure_name", "A0", "q", "w0_THz", "Gamma_THz"])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
