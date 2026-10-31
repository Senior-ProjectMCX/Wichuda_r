"""Semi-infinite correlation-diffusion BFI fitting (Ono et al., BOE 2023, Eq. 3).

The model estimates one effective BFI per SDS, not separate layer BFIs.
BFI is reported in units of 1e-9 cm^2/s, consistent with the
reference dataset's numerical convention (verify against the paper).
"""
import numpy as np
from scipy.optimize import least_squares
from parameters import MU_A_CM, MU_SP_CM, WAVELENGTH_NM, REFRACTIVE_INDEX_N

BFI_UNIT_CM2_S = 1e-9


def theoretical_g2(tau_us, sds_cm, bfi_units, beta):
    """Normalized semi-infinite correlation-diffusion g2."""
    tau = np.asarray(tau_us, dtype=float) * 1e-6
    rho = float(sds_cm)
    mus = float(MU_SP_CM)
    mua = float(MU_A_CM)
    k0 = 2.0 * np.pi * REFRACTIVE_INDEX_N / (WAVELENGTH_NM * 1e-7)
    # Diffuse extrapolated-boundary approximation; n is medium/air ratio.
    n = REFRACTIVE_INDEX_N
    reff = -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n
    z0 = 1.0 / mus
    zb = 2.0 * (1.0 - reff) / (3.0 * mus * (1.0 + reff))
    r1 = np.hypot(rho, z0)
    r2 = np.hypot(rho, z0 + 2.0 * zb)
    kd = np.sqrt(3.0 * mus * mua + 6.0 * mus**2 * k0**2 * bfi_units * BFI_UNIT_CM2_S * tau)
    gd = np.exp(-kd * r1) / r1 - np.exp(-kd * r2) / r2
    g0 = np.exp(-kd[0] * r1) / r1 - np.exp(-kd[0] * r2) / r2 if tau[0] == 0 else (
        np.exp(-np.sqrt(3.0 * mus * mua) * r1) / r1 - np.exp(-np.sqrt(3.0 * mus * mua) * r2) / r2
    )
    return 1.0 + beta * (gd / g0) ** 2


def fit_bfi(g2, sds_cm, tau_us=None):
    """Fit effective BFI and beta from g2, using tau > 0 to avoid g2(0) artifact."""
    y = np.asarray(g2, dtype=float)
    tau = np.arange(len(y), dtype=float) if tau_us is None else np.asarray(tau_us, dtype=float)
    mask = (tau > 0) & np.isfinite(y) & (y > 0)
    if mask.sum() < 20:
        raise ValueError('Not enough valid g2 points for BFI fit')
    x, yy = tau[mask], y[mask]
    beta0 = float(np.clip(np.percentile(yy[:min(10, len(yy))] - 1, 75), 0.02, 0.95))
    def residual(p):
        return theoretical_g2(x, sds_cm, p[0], p[1]) - yy
    fit = least_squares(residual, x0=[10.0, beta0], bounds=([0.0, 0.001], [100000.0, 1.5]),
                        max_nfev=180, xtol=1e-10, ftol=1e-10)
    return {'bfi': float(fit.x[0]), 'beta': float(fit.x[1]),
            'fit_rmse': float(np.sqrt(np.mean(fit.fun**2))), 'success': bool(fit.success)}


def bfi_comparison(results, trial_indices, dataset):
    """Compare model g2-derived BFI with mean measured BFI for selected physical trials."""
    out = {}
    for sds in (1, 2, 3):
        estimated = fit_bfi(results[sds]['generated_g2'], sds)
        physical_bfi = np.asarray(dataset[3 + sds, trial_indices], dtype=float)
        physical_bfi = physical_bfi[np.isfinite(physical_bfi)]
        if not len(physical_bfi):
            raise ValueError(f'No reference BFI for SDS {sds}')
        reference = float(np.mean(physical_bfi))
        error = abs(estimated['bfi'] - reference) / abs(reference) * 100 if reference != 0 else np.nan
        out[sds] = {'reference_bfi': reference, 'estimated_bfi': estimated['bfi'],
                    'error_percent': float(error), 'fit_rmse': estimated['fit_rmse'],
                    'reference_std': float(np.std(physical_bfi)), 'n_trials': len(physical_bfi)}
    return out
