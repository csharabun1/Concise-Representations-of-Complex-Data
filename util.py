"""
Shared helpers for the DDQ inference workflow.

The notebook pulls all constants, physics utilities, ARTS wrappers, caching
helpers, and evaluation routines from this module to stay uncluttered.
"""

from contextlib import contextmanager
import os

import cvxpy as cp
import numpy as np
import pyarts
import scipy
from scipy import constants as _C
from scipy.sparse import diags
import xarray as xr

from FluxSimulator import generate_gridded_field_from_profiles

# -------------------------------------------------------------------
# Constants and global knobs used across the workflow
# -------------------------------------------------------------------
D = 2.0
g = 9.81
N_A = _C.N_A
constant_k = _C.k

M_DRY = 28.9647e-3
MOLAR_MASS = {"CO2": 44.0095e-3, "O3": 47.9982e-3}

SPECIES = ["CO2", "O3"]
FORCING_SPECIES_ORDER = ["CO2", "CH4", "N2O", "O3", "CFC11", "CFC12"]

DEFAULT_F_ARRAY = np.array([0.15, 1, 1, 1, 1, 1, 1, 1], dtype=float)

# Globals used by cost_no_forcing; set via set_cost_dimensions / context
N_LEVELS = None
N_COLS = None
N_SCENARIOS = None
F_ARRAY = DEFAULT_F_ARRAY.copy()
mat = None


class _Temp1D:
    """Tiny adapter so flux_up/down can use len(temp) and temp.values[...]"""

    __slots__ = ("values",)

    def __init__(self, arr):
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1:
            raise ValueError(f"temperature must be 1D; got {a.shape}")
        self.values = a

    def __len__(self):
        return self.values.shape[0]


# -------------------------------------------------------------------
# Array + unit helpers
# -------------------------------------------------------------------
def B_nu(T, nu):
    """
    Planck function.

    In:
      T [K]: temperature
      nu [cm-1]: multiply all nu's by 100 to convert to m-1 in formula,
                 then multiply B by 100 for units of W/m^2/sr/cm-1
    Out:
      Planck function in units of W/m^2/sr/cm-1
    """
    k_B = scipy.constants.k  # Boltzmann constant
    h = scipy.constants.h  # Planck constant
    c = scipy.constants.speed_of_light  # speed of light in a vacuum

    return ((2 * h * (c**2) * ((100 * nu) ** 3)) / (np.exp((h * c * (100 * nu)) / (k_B * T)) - 1)) * 100


def squeeze_1d(a):
    out = np.asarray(a).squeeze()
    if out.ndim != 1:
        raise ValueError(f"Expected 1D after squeeze; got {out.shape}")
    return out


def assert_strictly_decreasing(vec, name="vector"):
    v = squeeze_1d(vec).astype(float)
    if not np.all(np.diff(v) < 0):
        raise ValueError(f"{name} must be strictly decreasing; got first/last = {v[0]}, {v[-1]}")


def vmr_to_mmr(vmr_species, molar_mass_species, molar_mass_dry_air=M_DRY):
    vmr = squeeze_1d(vmr_species).astype(float)
    return vmr * (float(molar_mass_species) / float(molar_mass_dry_air))


def absorption_coeff_to_xsec(k_species, p_level, T_level, vmr_species):
    """
    k_s [m^-1] -> σ_s [m^2/molecule], N_s = p*VMR/(k_B*T)
    k_species: (n_wvn, n_level); p,T,vmr: (n_level,)
    """
    p = squeeze_1d(p_level).astype(float)[None, :]
    T = squeeze_1d(T_level).astype(float)[None, :]
    vmr = squeeze_1d(vmr_species).astype(float)[None, :]
    N_s = (p * vmr) / (constant_k * T)
    return np.asarray(k_species, float) / np.clip(N_s, 1e-300, None)


def xsec_to_mass_absorption(sigma_s, molar_mass_species):
    """σ_s [m^2/molecule] -> κ_m [m^2/kg]"""
    return (N_A / float(molar_mass_species)) * np.asarray(sigma_s, float)


# -------------------------------------------------------------------
# Radiative-transfer core
# -------------------------------------------------------------------
def tau_from_linear_k(k_level_wvn, z_levels_m, center="level"):
    """
    Level-centered linear extinction k [m^-1] + geometric Δz -> layer τ (dimensionless).
    k_level_wvn : (n_level, n_wvn)
    z_levels_m  : (n_level,)
    returns: tau_layer (n_layer, n_wvn), tau_cum_top (n_layer, n_wvn)
    """
    z = squeeze_1d(z_levels_m).astype(float)
    k = np.asarray(k_level_wvn, dtype=float)
    if k.shape[0] != z.size:
        raise ValueError(f"Level mismatch: k has {k.shape[0]} levels, z has {z.size}")
    dz = np.abs(np.diff(z))  # (n_layer,)
    if center == "level":
        k_layer = 0.5 * (k[:-1, :] + k[1:, :])  # (n_layer, n_wvn)
    elif center == "layer":
        if k.shape[0] != dz.size:
            raise ValueError("For center='layer', k must be (n_layer, n_wvn)")
        k_layer = k
    else:
        raise ValueError("center must be 'level' or 'layer'")
    tau_layer = k_layer * dz[:, None]  # dimensionless
    top_to_surface = z[0] > z[-1]
    tau_for_csum = tau_layer if top_to_surface else tau_layer[::-1, :]
    tau_cum_top = np.cumsum(tau_for_csum, axis=0)
    tau_cum_top = tau_cum_top if top_to_surface else tau_cum_top[::-1, :]
    return tau_layer, tau_cum_top


def flux_up(tau, temperature, fgrid):
    """
    For arrays ordered TOA to surface, take temperature on half-levels and optical depth on full levels and compute flux

    The function is fragile/janky: it assumes a single profile, and assumes a (vertical) ordering, and ignores
    xarray dimension/coordinate names.
    """
    assert tau.shape[1] + 1 == len(temperature)
    D_local = 2  # Diffusivity factor - use what ARTS uses (Gaussian integration)
    F = np.zeros(shape=(tau.shape[0], len(temperature)))
    F[:, -1] = B_nu(temperature.values[-1], fgrid)
    for i in np.arange(len(temperature) - 2, -1, -1):
        S = B_nu(temperature.values[i], fgrid) * (1 - np.exp(-D_local * tau[:, i]))
        F[:, i] = F[:, i + 1] * np.exp(-D_local * tau[:, i]) + S

    return np.pi * F


def flux_down(tau, temperature, fgrid):
    assert tau.shape[1] + 1 == len(temperature)
    D_local = 2  # Diffusivity factor - use what ARTS uses (Gaussian integration)
    F = np.zeros(shape=(tau.shape[0], len(temperature)))
    F[:, 0] = 0.0
    for i in range(tau.shape[1]):
        S = B_nu(temperature.values[i + 1], fgrid) * (1.0 - np.exp(-D_local * tau[:, i]))
        F[:, i + 1] = F[:, i] * np.exp(-D_local * tau[:, i]) + S

    return np.pi * F


def broadband_flux_from_two_stream_tau(tau_layer_wvn, T_hl_K, nu_cm1, weights, to_surface_order=True):
    """
    Use the existing two-stream routines (flux_up, flux_down) to compute broadband profiles.

    Inputs:
      tau_layer_wvn : (n_wvn, n_layer)   τ for each ν and layer (surface→TOA or TOA→surface; see below)
      T_hl_K        : (n_level,)         temperatures on half-levels
      nu_cm1        : (n_wvn,)
      weights       : (n_wvn,)           DDQ spectral weights
      to_surface_order : bool            If True, return profiles in surface→TOA order (match cost)
    Returns:
      F_up, F_down, F_net : (n_level,) broadband W/m^2, ordered per to_surface_order
    """
    tau_ts = tau_layer_wvn[:, ::-1]
    T_ts = np.asarray(T_hl_K, float)[::-1]

    n_wvn = tau_ts.shape[0]
    n_lev = T_ts.size
    assert tau_ts.shape[1] + 1 == n_lev, (
        f"tau has {tau_ts.shape[1]} layers, but T has {n_lev} levels; need n_levels = n_layers + 1"
    )
    assert n_wvn == len(nu_cm1) == len(weights), (
        f"ν/weight mismatch: tau has {n_wvn} spectra, |nu|={len(nu_cm1)}, |W|={len(weights)}"
    )

    T_obj = _Temp1D(T_ts)

    Fu_spec = flux_up(tau_ts, T_obj, nu_cm1)  # (n_wvn, n_level)
    Fd_spec = flux_down(tau_ts, T_obj, nu_cm1)  # (n_wvn, n_level)

    w = np.asarray(weights, float).reshape(-1, 1)
    Fu = np.sum(w * Fu_spec, axis=0)
    Fd = np.sum(w * Fd_spec, axis=0)
    Fnet = Fu - Fd

    if to_surface_order:
        Fu, Fd, Fnet = Fu[::-1], Fd[::-1], Fnet[::-1]
    return Fu, Fd, Fnet


# -------------------------------------------------------------------
# κ_m evaluation helpers
# -------------------------------------------------------------------
def eval_kappam_levelwise_logpoly6(coeffs_wvn_6, p_levels_Pa, T_levels_K):
    """
    Evaluate κ_m(ν; T,p) at column levels for ALL ν at once.
    coeffs_wvn_6 : (n_wvn, 6)  [β0, β1 ln p, β2 ln T, β3 (ln p)^2, β4 (ln T)^2, β5 ln p ln T]
    p_levels_Pa  : (n_level,)  strictly decreasing (surface→TOA)
    T_levels_K   : (n_level,)
    Returns: κ_m  (n_wvn, n_level) in m^2/kg
    """
    p = np.asarray(p_levels_Pa, float)
    T = np.asarray(T_levels_K, float)
    lp, lt = np.log(p), np.log(T)  # (n_level,)
    X = np.stack([np.ones_like(lp), lp, lt, lp**2, lt**2, lp * lt], axis=1)  # (n_level, 6)
    return np.exp(X @ coeffs_wvn_6.T).T


def kappa_m_to_linear(kappa_m_s, p_level, T_level, mmr_species, R=287.0):
    """
    κ_m,s [m^2/kg] + species mass density ρ_s -> linear k_s [m^-1]
    ρ_air = p / (R * T);  ρ_s = q_s * ρ_air;  k_s = κ_m,s * ρ_s
    Broadcast over (n_wvn, n_level)
    """
    p = squeeze_1d(p_level).astype(float)
    T = squeeze_1d(T_level).astype(float)
    q = squeeze_1d(mmr_species).astype(float)
    rho_air = (p / (R * T))[None, :]
    rho_s = q[None, :] * rho_air
    return np.asarray(kappa_m_s, dtype=float) * rho_s


# -------------------------------------------------------------------
# Fitting utilities (log-polynomial κ_m models and diagnostics)
# -------------------------------------------------------------------
def logpoly_features_mesh(p_Pa, T_K):
    """
    Build features on a full mesh (n_p, n_T) → (n_p*n_T, 6):
    [1, ln p, ln T, (ln p)^2, (ln T)^2, (ln p)(ln T)]
    """
    lp = np.log(np.asarray(p_Pa).reshape(-1, 1))
    lt = np.log(np.asarray(T_K).reshape(1, -1))
    LP = np.repeat(lp, lt.shape[1], axis=1)
    LT = np.repeat(lt, lp.shape[0], axis=0)
    X = np.stack([np.ones_like(LP), LP, LT, LP**2, LT**2, LP * LT], axis=-1)
    return X.reshape(-1, 6)


def fit_logpoly_per_wavenumber(kappa_m_wvn_pT, p_grid_Pa, T_set_K, ridge=1e-6):
    """
    kappa_m_wvn_pT: (n_wvn, n_p, n_T)  → coeffs: (n_wvn, 6)
    """
    n_wvn, _, _ = kappa_m_wvn_pT.shape
    X = logpoly_features_mesh(p_grid_Pa, T_set_K)
    Xt = X.T
    G = Xt @ X
    G.flat[::7] += ridge
    Ginv_Xt = np.linalg.solve(G, Xt)
    coeffs = np.empty((n_wvn, 6), dtype=float)
    for i in range(n_wvn):
        y = np.log(np.clip(kappa_m_wvn_pT[i].reshape(-1), 1e-300, None))
        coeffs[i] = Ginv_Xt @ y
    return coeffs


def eval_logpoly_on_grid(coeffs_i, p_grid_Pa, T_set_K):
    """
    Evaluate a single-ν model on full (p×T) grid → (n_p, n_T)
    """
    lp = np.log(np.asarray(p_grid_Pa).reshape(-1, 1))
    lt = np.log(np.asarray(T_set_K).reshape(1, -1))
    X = np.stack(
        [
            np.ones_like(lp @ np.ones_like(lt)),
            np.repeat(lp, lt.shape[1], axis=1),
            np.repeat(lt, lp.shape[0], axis=0),
            (np.repeat(lp, lt.shape[1], axis=1)) ** 2,
            (np.repeat(lt, lp.shape[0], axis=0)) ** 2,
            (np.repeat(lp, lt.shape[1], axis=1)) * (np.repeat(lt, lp.shape[0], axis=0)),
        ],
        axis=-1,
    )
    return np.exp((X @ coeffs_i).astype(float))


def metrics_kappam_fit(kappa_true, kappa_pred):
    eps = 1e-30
    diff = kappa_pred - kappa_true
    abs_rms = float(np.sqrt(np.mean(diff**2)))
    rel = np.abs(diff) / np.maximum(np.abs(kappa_true), eps)
    return {"abs_rms": abs_rms, "rel_median": float(np.nanmedian(rel)), "rel_p95": float(np.nanpercentile(rel, 95))}


LOGPOLY6_TERMS = ["1", "ln p", "ln T", "(ln p)^2", "(ln T)^2", "(ln p)(ln T)"]


def _coefs_orient_to_wvn_rows(coefs, nu_cm1):
    """
    Ensure coeffs are shaped (n_wvn, 6). Accepts (6, n_wvn) or (n_wvn, 6).
    """
    C = np.asarray(coefs)
    nnu = len(nu_cm1)
    if C.ndim != 2:
        raise ValueError(f"coeffs must be 2-D; got {C.shape}")
    if C.shape == (nnu, 6):
        return C
    if C.shape == (6, nnu):
        return C.T
    raise ValueError(f"Unexpected coeffs shape {C.shape}; expected (n_wvn,6) or (6,n_wvn) with n_wvn={nnu}")


def format_logpoly6_equation_row(a, nu, species, cfmt="{:+.6e}"):
    """
    a: length-6 array of coefficients [a0..a5]
    Returns a single Markdown-formatted string for this ν.
    """
    a0, a1, a2, a3, a4, a5 = [cfmt.format(x) for x in a]
    eq = (
        f"ln kappa_m(ν={nu:.2f} cm^-1; {species}) = "
        f"{a0} + {a1}·ln p + {a2}·ln T + {a3}·(ln p)^2 + {a4}·(ln T)^2 + {a5}·(ln p)(ln T)"
    )
    return eq


def print_logpoly6_equations(coeffs, nu_cm1, species, sample=None, to_file=None, cfmt="{:+.6e}"):
    """
    Pretty-print fitted log–poly equations per wavenumber.
    """
    C = _coefs_orient_to_wvn_rows(coeffs, nu_cm1)
    nnu = len(nu_cm1)

    if sample is None:
        idxs = np.arange(nnu)
    elif isinstance(sample, int):
        idxs = np.linspace(0, nnu - 1, sample, dtype=int)
    else:
        idxs = np.asarray(list(sample), dtype=int)

    lines = []
    lines.append(f"### Fitted log–polynomial for {species}\n")
    lines.append(
        "Model:  \n`ln kappa_m(ν;T,p) = a0 + a1 ln p + a2 ln T + a3 (ln p)^2 + a4 (ln T)^2 + a5 (ln p)(ln T)`  \n"
        "_Units: kappa_m in m^2/kg, p in Pa, T in K._\n"
    )
    for j in idxs:
        lines.append(f"- {format_logpoly6_equation_row(C[j], nu_cm1[j], species, cfmt=cfmt)}")

    md = "\n".join(lines)
    if to_file:
        with open(to_file, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    else:
        print(md)


# -------------------------------------------------------------------
# Fitting utilities (PLS surrogate on κ_m)
# -------------------------------------------------------------------
# Default wavenumber-index partitions used in linear_regressions_indiv_wvn.ipynb.
# These are indices into the DDQ grid (ddq.S), not physical wavenumbers.
PLS_WVN_GROUPS = {
    # NOTE: The original notebook had CO2 index 20 duplicated across the lists and omitted 26.
    # We use a disjoint, full-cover partition here.
    "CO2": {
        "zeros": list(range(17)) + list(range(43, 64)),
        "by_power": {
            1: [20, 21, 22, 23, 24, 25, 28, 31, 32, 33, 34, 35, 38, 40],
            3: [17, 18, 19, 26, 27, 29, 30, 36, 37, 39, 41, 42],
        },
        "n_components": 2,
    },
    "O3": {
        "zeros": list(range(17)) + list(range(43, 64)),
        "by_power": {
            1: [17, 18, 19, 20, 21, 22, 24, 31, 32, 34, 35, 39, 40],
            2: [23, 25, 26, 27, 28, 29, 30, 33, 36, 37, 38, 41, 42],
        },
        "n_components": 2,
    },
}


def _validate_wvn_partition(n_wvn, wvn_zeros, wvn_by_power):
    zeros = list(map(int, wvn_zeros))
    by_power = {int(p): list(map(int, idxs)) for p, idxs in wvn_by_power.items()}

    all_idxs = zeros[:]
    for idxs in by_power.values():
        all_idxs.extend(idxs)

    # Range check
    bad = [i for i in all_idxs if i < 0 or i >= n_wvn]
    if bad:
        raise ValueError(f"Wavenumber indices out of range 0..{n_wvn-1}: {sorted(set(bad))}")

    # Duplicate check across groups
    seen = set()
    dup = set()
    for i in all_idxs:
        if i in seen:
            dup.add(i)
        seen.add(i)

    missing = sorted(set(range(n_wvn)) - set(all_idxs))
    if dup or missing:
        raise ValueError(
            "Invalid DDQ index partition. "
            f"duplicates={sorted(dup) if dup else []}, missing={missing if missing else []}"
        )


def _pls_tp_features(T, P, power):
    """
    Build PLS features from temperature and pressure vectors.
    power=1 -> [T, P]
    power>1 -> [1, T^1, P^1, ..., T^power, P^power]
    """
    T = np.asarray(T, float).ravel()
    P = np.asarray(P, float).ravel()
    if T.shape != P.shape:
        raise ValueError(f"T and P must have same shape; got {T.shape} vs {P.shape}")

    if power == 1:
        return np.stack([T, P], axis=1)

    ones = np.ones_like(T)
    X = np.empty((T.size, int(power) * 2 + 1), dtype=float)
    X[:, 0] = ones
    k = 1
    for j in range(1, int(power) + 1):
        X[:, k] = T**j
        k += 1
        X[:, k] = P**j
        k += 1
    return X


def train_pls_kappa_arrays(
    kappa_train_wvn_pT,
    P_train_Pa,
    T_train_K,
    test_pressure_Pa,
    test_temperature_K,
    wvn_zeros,
    wvn_by_power,
    n_components=2,
    log10_floor=1e-30,
):
    """
    Train per-wavenumber PLS regressions on κ_m(T,p) training grid and predict κ_m for a set of columns.

    Inputs:
      kappa_train_wvn_pT : (n_wvn, n_p, n_T) true κ_m samples [m^2/kg]
      P_train_Pa         : (n_p,) training pressures [Pa]
      T_train_K          : (n_T,) training temperatures [K]
      test_pressure_Pa   : (n_col, n_level) column pressures [Pa]
      test_temperature_K : (n_col, n_level) column temperatures [K]
      wvn_zeros          : list[int] wvn indices to force κ_m=0
      wvn_by_power       : dict[int, list[int]] mapping polynomial power -> wvn indices
      n_components       : PLS components (default 2)
      log10_floor        : floor for log10 transform to avoid -inf

    Returns:
      kappa_pred_wvn_col_lev : (n_wvn, n_col, n_level) κ_m predictions [m^2/kg]
    """
    from sklearn.cross_decomposition import PLSRegression

    kappa_grid = np.asarray(kappa_train_wvn_pT, float)
    if kappa_grid.ndim != 3:
        raise ValueError(f"kappa_train_wvn_pT must be 3D (n_wvn,n_p,n_T); got {kappa_grid.shape}")

    n_wvn, n_p, n_T = kappa_grid.shape
    P_train = np.asarray(P_train_Pa, float).reshape(-1)
    T_train = np.asarray(T_train_K, float).reshape(-1)
    if P_train.size != n_p or T_train.size != n_T:
        raise ValueError(f"Training grid mismatch: kappa has (n_p={n_p}, n_T={n_T}) but P={P_train.size}, T={T_train.size}")

    _validate_wvn_partition(n_wvn, wvn_zeros, wvn_by_power)

    test_P = np.asarray(test_pressure_Pa, float)
    test_T = np.asarray(test_temperature_K, float)
    if test_P.shape != test_T.shape or test_P.ndim != 2:
        raise ValueError(f"test_pressure_Pa and test_temperature_K must both be (n_col,n_level); got {test_P.shape} and {test_T.shape}")

    n_col, n_level = test_P.shape
    X_test = None  # built per power

    kappa_pred = np.zeros((n_wvn, n_col, n_level), dtype=float)

    # Precompute training mesh vectors once
    TT, PP = np.meshgrid(T_train, P_train)  # (n_p, n_T)
    T_mesh = TT.ravel()
    P_mesh = PP.ravel()

    # Train/predict for each power group
    for power, wvn_list in wvn_by_power.items():
        wvn_list = list(map(int, wvn_list))
        if not wvn_list:
            continue

        X_train = _pls_tp_features(T_mesh, P_mesh, power=power)
        X_test = _pls_tp_features(test_T.ravel(), test_P.ravel(), power=power)

        if n_components > X_train.shape[1]:
            raise ValueError(f"n_components={n_components} > n_features={X_train.shape[1]} for power={power}")

        for wi in wvn_list:
            y = np.log10(np.clip(kappa_grid[wi, :, :].ravel(), log10_floor, None))
            model = PLSRegression(int(n_components))
            model.fit(X_train, y.reshape(-1, 1))
            yhat = model.predict(X_test).reshape(n_col, n_level)
            kappa_pred[wi, :, :] = np.power(10.0, yhat)

    # Enforce exact zeros for designated wavenumbers
    for wi in map(int, wvn_zeros):
        kappa_pred[wi, :, :] = 0.0

    return kappa_pred


# -------------------------------------------------------------------
# Profile utilities
# -------------------------------------------------------------------
def ensure_surface_to_toa_order(profile, species_order):
    """
    Return p_hl, T_hl, vmr (dict) all sorted to surface→TOA (strictly decreasing p).
    Does NOT mutate `profile`.
    """
    p_hl = np.asarray(profile.pressure_hl.values, float)
    T_hl = np.asarray(profile.temperature_hl.values, float)

    vmr = {}
    for s in species_order:
        key = f"{s.lower()}_mole_fraction_hl"
        vmr[s] = np.asarray(profile[key].values, float)

    if not np.all(np.diff(p_hl) < 0):
        ord_idx = np.argsort(p_hl)[::-1]
        p_hl = p_hl[ord_idx]
        T_hl = T_hl[ord_idx]
        for s in species_order:
            vmr[s] = vmr[s][ord_idx]

    return p_hl, T_hl, vmr


def hypsometric_heights_from_pT(p_hl_Pa, T_hl_K, z0=0.0, R_dry=287.0, g=9.81):
    """
    Hypsometric integration on half-levels (surface→TOA, strictly decreasing p).
    Returns z_hl [m] with z[0]=z0 at surface.
    """
    p = np.asarray(p_hl_Pa, float)
    T = np.asarray(T_hl_K, float)
    if not np.all(np.diff(p) < 0):
        raise ValueError("p_hl_Pa must be strictly decreasing (surface→TOA).")
    z = np.empty_like(p, dtype=float)
    z[0] = z0
    for i in range(1, p.size):
        Tbar = 0.5 * (T[i - 1] + T[i])
        z[i] = z[i - 1] + (R_dry * Tbar / g) * np.log(p[i - 1] / p[i])
    return z


def make_isothermal_atmosphere_for_species(p_hl_Pa, T_const_K, species, vmr_value):
    """
    Build a 1D isothermal column at pressure half-levels p_hl_Pa with constant T,
    only the target species present (others zero). p_hl must be strictly decreasing.
    """
    p_hl = np.asarray(p_hl_Pa, float)
    if not np.all(np.diff(p_hl) < 0):
        p_hl = p_hl[::-1]
    assert_strictly_decreasing(p_hl, "p_hl_Pa")

    T_hl = np.full_like(p_hl, float(T_const_K))
    gases = {species: np.full_like(p_hl, float(vmr_value))}
    atmosphere = generate_gridded_field_from_profiles(p_hl, T_hl, gases=gases, particulates={}, z_field=None)
    return atmosphere


def generate_kappam_grid_for_species(species, wvn_cm1, p_hl_grid_Pa, T_set_K, vmr_value):
    """
    Returns:
      kappa_m : (n_wvn, n_p, n_T)  on the actual ARTS level grid
      P_used  : (n_p,) [Pa]        strictly decreasing
      T_used  : (n_T,) [K]
    """
    n_wvn = len(wvn_cm1)
    n_T = len(T_set_K)
    kappa_list = []
    P_used = None

    for Tj in T_set_K:
        atm = make_isothermal_atmosphere_for_species(p_hl_grid_Pa, Tj, species, vmr_value)
        _, k_species_all, p_levels, _, _ = calc_lbl_rt(atm, species=(species,), w_grid=wvn_cm1)

        k_s = k_species_all[0, :, :]  # (n_wvn, n_level)
        Tlev = np.full_like(p_levels, float(Tj))
        VMR = np.full_like(p_levels, float(vmr_value))

        sigma = absorption_coeff_to_xsec(k_s, p_levels, Tlev, VMR)
        kappa = xsec_to_mass_absorption(sigma, MOLAR_MASS[species])
        kappa_list.append(kappa)

        if P_used is None:
            P_used = p_levels
        else:
            if len(p_levels) != len(P_used) or np.max(np.abs(p_levels - P_used)) > 1e-6:
                raise RuntimeError("ARTS vertical grid changed across temperatures; ensure consistent p-grid.")

    kappa_arr = np.stack(kappa_list, axis=0)  # (n_T, n_wvn, n_level)
    kappa_arr = np.moveaxis(kappa_arr, 0, 2)  # (n_wvn, n_level, n_T)
    return kappa_arr, P_used, np.asarray(T_set_K, float)


def derivative_matrix(nx):
    """
    Constructs the centered second-order accurate first-order derivative, equivalent to np.gradient.
    """
    diagonals = [[-1.0 / 2.0], [0], [1.0 / 2.0]]
    offsets = [-1, 0, 1]

    d1mat = diags(diagonals, offsets, shape=(nx, nx)).toarray()

    d1mat[0, :2] = np.array([-1, 1])
    d1mat[-1, -2:] = np.array([1, -1])

    return d1mat


def set_cost_dimensions(n_levels, n_cols, n_scenarios=1, f_array=None):
    """
    Update globals that cost_no_forcing expects.
    """
    global N_LEVELS, N_COLS, N_SCENARIOS, mat, F_ARRAY
    N_LEVELS = int(n_levels)
    N_COLS = int(n_cols)
    N_SCENARIOS = int(n_scenarios)
    mat = derivative_matrix(N_LEVELS)
    if f_array is not None:
        F_ARRAY = np.asarray(f_array, dtype=float)


def cost_no_forcing(y_hat, y_ref, x_sup):
    """
    In:
        y_hat: estimate; flattened array of (rows*scenarios*cols + gases*cols) shape,
            where the last columns are forcings
        y_ref: reference values/data
        x_sup: supplementary data, here, heating rates and pressures.
    Out:
        cost_function: the value of the cost function
    """
    if any(v is None for v in (N_LEVELS, N_COLS, N_SCENARIOS, mat, F_ARRAY)):
        raise RuntimeError("Cost dimensions are not set. Call set_cost_dimensions first.")

    if isinstance(y_hat, xr.Dataset):
        return cp.norm(y_ref.reference_forcing_co2.data - y_hat.reference_forcing_co2.data)

    g_local = 9.81
    scaling = 3600 * 24

    scenario_cols = np.random.randint(0, N_SCENARIOS, N_COLS, dtype=int) + N_SCENARIOS * np.arange(N_COLS)

    end_fluxes_idx = N_LEVELS * N_COLS * N_SCENARIOS
    end_co2_idx = end_fluxes_idx + N_COLS * N_SCENARIOS
    end_ch4_idx = end_co2_idx + N_COLS * N_SCENARIOS
    end_n2o_idx = end_ch4_idx + N_COLS * N_SCENARIOS
    end_o3_idx = end_n2o_idx + N_COLS * N_SCENARIOS
    end_cfc11_idx = end_o3_idx + N_COLS * N_SCENARIOS

    F_est = y_hat[:end_fluxes_idx]
    F_est = cp.reshape(F_est, [N_COLS * N_SCENARIOS, N_LEVELS], "F")[scenario_cols, :]
    F_est = cp.reshape(F_est, [N_COLS * N_LEVELS], "F")

    Force_est_co2 = y_hat[end_fluxes_idx:end_co2_idx][scenario_cols]
    Force_est_ch4 = y_hat[end_co2_idx:end_ch4_idx][scenario_cols]
    Force_est_n2o = y_hat[end_ch4_idx:end_n2o_idx][scenario_cols]
    Force_est_o3 = y_hat[end_n2o_idx:end_o3_idx][scenario_cols]
    Force_est_cfc11 = y_hat[end_o3_idx:end_cfc11_idx][scenario_cols]
    Force_est_cfc12 = y_hat[end_cfc11_idx:][scenario_cols]

    y_ref = y_ref.isel(column=scenario_cols)
    x_sup = x_sup.isel(column=scenario_cols)

    F_err = cp.norm((F_est - y_ref.reference_fluxes.data.reshape(-1)))

    H_est = -scaling * (cp.matmul(mat, cp.reshape(F_est, [N_COLS, N_LEVELS], "F").T) * g_local) / cp.matmul(mat, x_sup.pressures.data.T) / 1004

    H_err = cp.norm((H_est.T - x_sup.reference_heating))

    Force_err_co2 = cp.norm((Force_est_co2 - y_ref.reference_forcing_co2.data.reshape(-1)))
    Force_err_ch4 = cp.norm((Force_est_ch4 - y_ref.reference_forcing_ch4.data.reshape(-1)))
    Force_err_n2o = cp.norm((Force_est_n2o - y_ref.reference_forcing_n2o.data.reshape(-1)))
    Force_err_o3 = cp.norm((Force_est_o3 - y_ref.reference_forcing_o3.data.reshape(-1)))
    Force_err_cfc11 = cp.norm((Force_est_cfc11 - y_ref.reference_forcing_cfc11.data.reshape(-1)))
    Force_err_cfc12 = cp.norm((Force_est_cfc12 - y_ref.reference_forcing_cfc12.data.reshape(-1)))

    cost_function = (
        F_ARRAY[0] * F_err
        + F_ARRAY[1] * H_err
        + F_ARRAY[2] * Force_err_co2
        + F_ARRAY[3] * Force_err_ch4
        + F_ARRAY[4] * Force_err_n2o
        + F_ARRAY[5] * Force_err_o3
        + F_ARRAY[6] * Force_err_cfc11
        + F_ARRAY[7] * Force_err_cfc12
    )

    return cost_function


# -------------------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------------------
def ckdmip_envelope(ds):
    P = ds["pressure_hl"].values
    T = ds["temperature_hl"].values
    p_all = np.asarray(P, float).ravel()
    t_all = np.asarray(T, float).ravel()
    p_all = p_all[np.isfinite(p_all) & (p_all > 0.0)]
    t_all = t_all[np.isfinite(t_all)]
    return float(p_all.min()), float(p_all.max()), float(t_all.min()), float(t_all.max())


def make_sampling_grids_from_envelope(
    ds,
    nT=9,
    nP=55,
    T_pad=0.08,
    p_pad=0.08,
    p_top_floor=1e-2,
):
    """
    Build:
      - T_set: ascending, padded by T_pad fraction
      - p_hl_grid: strictly decreasing (surface→TOA), padded by p_pad fraction
    """
    p_min, p_max, t_min, t_max = ckdmip_envelope(ds)

    T_lo = max(t_min * (1.0 - T_pad), 100.0)
    T_hi = t_max * (1.0 + T_pad)
    T_set = np.linspace(T_lo, T_hi, int(nT))

    p_surf = p_max * (1.0 + p_pad)
    p_top = max(p_min * (1.0 - p_pad), p_top_floor)

    p_hl_grid = np.geomspace(p_top, p_surf, int(nP))[::-1]

    for i in range(1, p_hl_grid.size):
        if not (p_hl_grid[i] < p_hl_grid[i - 1]):
            p_hl_grid[i] = np.nextafter(p_hl_grid[i - 1], 0.0)

    if np.any(T_set <= 0.0):
        raise ValueError("T_set contains non-positive temperatures. Use fractional pads (e.g., 0.05).")
    if not np.all(np.diff(p_hl_grid) < 0.0):
        raise ValueError("p_hl_grid must be strictly decreasing (surface→TOA).")

    return T_set, p_hl_grid


def report_sampling_vs_envelope(ds, T_set, p_hl_grid, p_top_floor=1e-2, eps_frac=1e-3):
    p_min, p_max, t_min, t_max = ckdmip_envelope(ds)
    print(
        f"CKDMIP envelope: p∈[{p_min:.2e},{p_max:.2e}] Pa, T∈[{t_min:.1f},{t_max:.1f}] K\n"
        f"Sampling grids : p∈[{p_hl_grid.min():.2e},{p_hl_grid.max():.2e}] Pa (surf→TOA), "
        f"T∈[{T_set.min():.1f},{T_set.max():.1f}] K"
    )

    cover_p_hi = p_hl_grid.max() >= p_max * (1 - eps_frac)
    cover_p_lo = p_hl_grid.min() <= max(p_min, p_top_floor) * (1 + eps_frac)
    cover_T_lo = T_set.min() <= t_min * (1 + eps_frac)
    cover_T_hi = T_set.max() >= t_max * (1 - eps_frac)

    outside_p_hi = p_hl_grid.max() > p_max * (1 + eps_frac)
    outside_p_lo = (p_hl_grid.min() < p_min * (1 - eps_frac)) or (p_hl_grid.min() < p_top_floor * (1 - eps_frac))
    outside_T_lo = T_set.min() < t_min * (1 - eps_frac)
    outside_T_hi = T_set.max() > t_max * (1 + eps_frac)

    if cover_p_hi and cover_p_lo and cover_T_lo and cover_T_hi:
        if any([outside_p_hi, outside_p_lo, outside_T_lo, outside_T_hi]):
            print("✅ Sampling covers CKDMIP and extends a little outside.")
        else:
            print("✅ Sampling exactly covers the CKDMIP envelope (no outside padding).")
    else:
        print("⚠️ Sampling does not fully cover the CKDMIP envelope. Increase padding or nT/nP.")


# -------------------------------------------------------------------
# Surrogate helpers
# -------------------------------------------------------------------
def coerce_coeffs_wvn_by_features(coeffs_raw, nu_used, nu_target):
    """
    Ensure coeffs are (n_wvn_target, n_features) and aligned to nu_target.
    - coeffs_raw: 2D array, either (n_wvn_used, n_features) or (n_features, n_wvn_used)
    - nu_used:    1D array of wavenumbers the coeffs were trained on
    - nu_target:  1D array of wavenumbers to evaluate on (e.g., ddq.S)
    """
    if nu_used is None:
        nu_used = nu_target

    C = np.asarray(coeffs_raw)
    if C.ndim != 2:
        raise ValueError(f"coeffs must be 2D, got {C.shape}")

    if C.shape[0] == len(nu_used):
        C_wf = C
    elif C.shape[1] == len(nu_used):
        C_wf = C.T
    else:
        raise ValueError(f"coeffs shape {C.shape} inconsistent with len(nu_used)={len(nu_used)}")

    if np.array_equal(nu_used, nu_target):
        return C_wf

    idx = np.array([np.argmin(np.abs(nu_used - v)) for v in nu_target], dtype=int)
    if not np.allclose(nu_used[idx], nu_target, rtol=0, atol=1e-6):
        raise ValueError("Coefficient ν grid does not match target DDQ ν grid within tolerance.")
    return C_wf[idx, :]


def make_y_hat_with_species_swaps(
    target_species_list,
    gas_names,
    k_species_lbl,
    p_hl,
    T_hl,
    z_levels,
    nu_cm1,
    weights,
    kappa_coeffs_dict,
    molar_mass_dict,
    vmr_dict,
    swap_mode="replace",
):
    """
    Build y_hat for the cost from log-poly coefficients.

    swap_mode:
      - "replace" (default): subtract the LBL linear-k contribution for each target species and
        add the surrogate-reconstructed linear-k (true species swap).
    Returns:
      y_hat: flattened vector [F_net(levels), zeros_for_forcings]
    """
    swap_mode = str(swap_mode).strip().lower()
    if swap_mode not in ("replace"):
        raise ValueError(f"swap_mode must be 'replace'; got {swap_mode!r}")

    k_total_wvn_lev = np.sum(k_species_lbl, axis=0).copy()

    target_upper = [s.upper() for s in target_species_list]

    name_to_idx = {g: i for i, g in enumerate(gas_names)}

    for s in target_upper:
        if s not in name_to_idx:
            raise ValueError(f"Target species {s} not in LBL gas list {gas_names}")
        if s not in kappa_coeffs_dict:
            raise ValueError(f"No κ_m coefficients provided for species {s}")
        if s not in molar_mass_dict:
            raise ValueError(f"No molar mass for species {s}")
        if s not in vmr_dict:
            raise ValueError(f"No VMR profile available for species {s}")

        s_idx = name_to_idx[s]
        if swap_mode == "replace":
            k_total_wvn_lev -= k_species_lbl[s_idx, :, :]

        kappa_m_s = eval_kappam_levelwise_logpoly6(kappa_coeffs_dict[s], p_hl, T_hl)
        q_s = vmr_to_mmr(vmr_dict[s], molar_mass_dict[s])
        k_rec_s = kappa_m_to_linear(kappa_m_s, p_hl, T_hl, q_s)

        k_total_wvn_lev += k_rec_s

    tau_layer_wvn, _ = tau_from_linear_k(k_total_wvn_lev.T, z_levels, center="level")
    tau_layer_wvn = tau_layer_wvn.T

    Fu, Fd, Fnet = broadband_flux_from_two_stream_tau(tau_layer_wvn, T_hl, nu_cm1, weights, to_surface_order=True)

    y_hat = np.concatenate([Fnet.reshape(-1), np.zeros(6)], axis=0)
    return y_hat


def make_y_hat_with_kappa_arrays(
    target_species_list,
    gas_names,
    k_species_lbl,
    p_hl,
    T_hl,
    z_levels,
    nu_cm1,
    weights,
    kappa_arrays_dict,
    molar_mass_dict,
    vmr_dict,
    col_idx,
    swap_mode="replace",
):
    """
    Build y_hat using precomputed κ_m arrays (per wavenumber, per column, per level).

    swap_mode:
      - "replace" (default): subtract the LBL linear-k contribution for each target species and
        add the surrogate-reconstructed linear-k (true species swap).
    """
    swap_mode = str(swap_mode).strip().lower()
    if swap_mode not in ("replace"):
        raise ValueError(f"swap_mode must be 'replace'; got {swap_mode!r}")

    k_total_wvn_lev = np.sum(k_species_lbl, axis=0).copy()
    target_upper = [s.upper() for s in target_species_list]
    name_to_idx = {g: i for i, g in enumerate(gas_names)}

    for s in target_upper:
        if s not in name_to_idx:
            raise ValueError(f"Target species {s} not in LBL gas list {gas_names}")
        if s not in kappa_arrays_dict:
            raise ValueError(f"No κ_m array provided for species {s}")
        if s not in molar_mass_dict:
            raise ValueError(f"No molar mass for species {s}")
        if s not in vmr_dict:
            raise ValueError(f"No VMR profile available for species {s}")

        s_idx = name_to_idx[s]
        if swap_mode == "replace":
            k_total_wvn_lev -= k_species_lbl[s_idx, :, :]

        q_s = vmr_to_mmr(vmr_dict[s], molar_mass_dict[s])
        kappa_m_s = kappa_arrays_dict[s][:, col_idx, :]  # (n_wvn, n_level)
        k_rec_s = kappa_m_to_linear(kappa_m_s, p_hl, T_hl, q_s)

        k_total_wvn_lev += k_rec_s

    tau_layer_wvn, _ = tau_from_linear_k(k_total_wvn_lev.T, z_levels, center="level")
    tau_layer_wvn = tau_layer_wvn.T

    Fu, Fd, Fnet = broadband_flux_from_two_stream_tau(tau_layer_wvn, T_hl, nu_cm1, weights, to_surface_order=True)
    y_hat = np.concatenate([Fnet.reshape(-1), np.zeros(6)], axis=0)
    return y_hat


def _discover_all_gases(profile):
    """
    Return sorted list of GAS NAMES (UPPERCASE) found in the CKDMIP column.
    We detect gases by *_mole_fraction_hl or *_mole_fraction_fl.
    """
    species = set()
    for v in profile.variables:
        if v.endswith("_mole_fraction_hl") or v.endswith("_mole_fraction_fl"):
            base = v.split("_mole_fraction")[0]
            species.add(base.upper())
    return sorted(species)


def trapezoid_weights(nu_cm1: np.ndarray) -> np.ndarray:
    """Nonnegative trapezoid bin widths on a strictly increasing wavenumber grid."""
    nu = np.asarray(nu_cm1, float)
    if not np.all(np.diff(nu) > 0):
        raise ValueError("nu_cm1 must be strictly increasing for trapezoid weights.")
    dnu = np.diff(nu)
    w = np.empty_like(nu)
    w[1:-1] = 0.5 * (dnu[:-1] + dnu[1:])
    w[0] = 0.5 * dnu[0]
    w[-1] = 0.5 * dnu[-1]
    return w


# -------------------------------------------------------------------
# LBL baselines
# -------------------------------------------------------------------
def calc_absorp_coeff(atmosphere, species=("CO2",), w_grid=None):
    """
    Returns:
      absorption_coeff : (n_species, n_wvn, n_level)  [1/m]
      p_levels         : (n_level,) [Pa]  strictly decreasing (surface→TOA)
      z_levels_hyp     : (n_level,) [m]   hypsometric from p & T
      T_levels         : (n_level,) [K]
    """
    assert w_grid is not None and len(w_grid) > 0
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, 0)
    ws.IndexSet(ws.stokes_dim, 1)
    ws.jacobianOff()
    ws.cloudboxOff()

    sp_list = list(species)
    ws.abs_speciesSet(species=sp_list)
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.ReadXsecData(basename="xsec/")

    ws.f_grid = pyarts.arts.convert.kaycm2freq(w_grid)
    ws.abs_lines_per_speciesCompact()
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesNormalization(option="SFS")
    ws.abs_lines_per_speciesTurnOffLineMixing()

    ws.propmat_clearsky_agendaAuto(T_extrapolfac=1e99)
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    ws.atm_fields_compact = atmosphere
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]
    ws.vmr_field.value = ws.vmr_field.value.value.clip(min=0.0)

    ws.cloudboxSetFullAtm()
    ws.scat_data_checked = 1
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()
    ws.sensorOff()
    ws.jacobianOff()

    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    ws.propmat_clearsky_fieldCalc()
    k_raw = np.asarray(ws.propmat_clearsky_field.value)

    shp = k_raw.shape
    nf = len(w_grid)
    nlev = int(np.asarray(ws.p_grid.value).size)
    ns = len(sp_list)

    cand_f = [i for i, s in enumerate(shp) if s == nf]
    cand_p = [i for i, s in enumerate(shp) if s == nlev]
    if len(cand_f) != 1 or len(cand_p) != 1:
        raise ValueError(f"Cannot identify freq/level axes from shape {shp} (nf={nf}, nlev={nlev}); found f={cand_f}, p={cand_p}")
    ax_f, ax_p = cand_f[0], cand_p[0]

    cand_s = [i for i, s in enumerate(shp) if s == ns]
    if ns > 1 and len(cand_s) == 1:
        ax_s = cand_s[0]
        k_re = np.moveaxis(k_raw, (ax_s, ax_f, ax_p), (0, 1, 2))
        k_re = np.asarray(k_re).squeeze()
        if k_re.ndim == 2:
            k_re = k_re[None, :, :]
        if k_re.shape[0] != ns or k_re.shape[1] != nf or k_re.shape[2] != nlev:
            raise ValueError(f"Unexpected reordered shape {k_re.shape}; expected (ns={ns}, nf={nf}, nlev={nlev})")
        absorption_coeff = k_re
    else:
        k_fp = np.moveaxis(k_raw, (ax_f, ax_p), (0, 1))
        k_fp = np.squeeze(k_fp)
        if k_fp.ndim != 2 or k_fp.shape != (nf, nlev):
            raise ValueError(f"Fallback expected (nf, nlev) after squeeze; got {k_fp.shape} from {shp}")
        absorption_coeff = k_fp[None, :, :]

    p_levels = np.asarray(ws.p_grid.value, dtype=float).ravel()
    T_levels = np.asarray(ws.t_field.value, dtype=float)[:, 0, 0]

    R = 287.0
    g0 = 9.81
    z_levels_hyp = np.zeros_like(p_levels, dtype=float)
    for i in range(1, p_levels.size):
        p1, p2 = p_levels[i - 1], p_levels[i]
        Tbar = 0.5 * (T_levels[i - 1] + T_levels[i])
        z_levels_hyp[i] = z_levels_hyp[i - 1] + (R * Tbar / g0) * np.log(p1 / p2)

    return absorption_coeff, p_levels, z_levels_hyp, T_levels


def build_reference_targets_from_lbl_dense(
    profile,
    col,
    nu_dense_cm1,
    N_LEVELS,
    mat,
    N_COLS=1,
    N_SCENARIOS=1,
    cache_dir="lbl_dense",
):
    profile = profile.isel(column=col)
    gas_names = _discover_all_gases(profile)
    if not gas_names:
        raise ValueError("No gas fields found in profile.")

    p_hl, T_hl, vmr_dict = ensure_surface_to_toa_order(profile, gas_names)
    if p_hl.size != N_LEVELS:
        raise ValueError(f"N_LEVELS={N_LEVELS} does not match column levels={p_hl.size}")

    atmosphere = generate_gridded_field_from_profiles(
        p_hl, T_hl, gases={g: vmr_dict[g] for g in gas_names}, particulates={}, z_field=None
    )

    weights_dense = trapezoid_weights(nu_dense_cm1)

    cache_filename = f"dense_lbl_col_{col:03d}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Loading dense LBL data for col {col} from cache: {cache_path}")
        with np.load(cache_path) as data:
            k_species_dense = data["k_species_dense"]
            p_levels = data["p_levels"]
            z_levels = data["z_levels"]
            T_levels = data["T_levels"]
    else:
        print(f"Calculating dense LBL data for col {col} and caching to: {cache_path}")

        k_species_dense, p_levels, z_levels, T_levels = calc_absorp_coeff(atmosphere, species=tuple(gas_names), w_grid=nu_dense_cm1)

        os.makedirs(cache_dir, exist_ok=True)

        np.savez_compressed(
            cache_path,
            k_species_dense=k_species_dense,
            p_levels=p_levels,
            z_levels=z_levels,
            T_levels=T_levels,
        )

    k_total_dense = np.sum(k_species_dense, axis=0)
    tau_layer_dense, _ = tau_from_linear_k(k_total_dense.T, z_levels)
    tau_layer_dense = tau_layer_dense.T

    Fu_ref, Fd_ref, Fnet_ref = broadband_flux_from_two_stream_tau(tau_layer_dense, T_hl, nu_dense_cm1, weights_dense, to_surface_order=True)

    F_ref_2D = np.tile(Fnet_ref.reshape(N_LEVELS, 1), (1, N_COLS))
    p_2D = np.tile(p_hl.reshape(N_LEVELS, 1), (1, N_COLS))
    H_ref_2D = -86400.0 * (mat @ F_ref_2D) * 9.8 / (mat @ p_2D) / 1004.0

    level = np.arange(N_LEVELS)
    column = np.arange(N_COLS)

    y_ref_dense = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_ref_2D),
            reference_forcing_co2=(("column",), np.zeros(N_COLS)),
            reference_forcing_ch4=(("column",), np.zeros(N_COLS)),
            reference_forcing_n2o=(("column",), np.zeros(N_COLS)),
            reference_forcing_o3=(("column",), np.zeros(N_COLS)),
            reference_forcing_cfc11=(("column",), np.zeros(N_COLS)),
            reference_forcing_cfc12=(("column",), np.zeros(N_COLS)),
        ),
        coords=dict(level=level, column=column),
    )

    x_sup_dense = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), p_2D.T),
            reference_heating=(("column", "level"), H_ref_2D.T),
        ),
        coords=dict(level=level, column=column),
    )

    return y_ref_dense, x_sup_dense, (gas_names, k_species_dense, p_hl, T_hl, z_levels, vmr_dict)


def calc_lbl_rt(atmosphere, species=("CO2",), w_grid=None):
    """
    Returns:
      tau_arts         : (n_wvn, n_layer)
      absorption_coeff : (n_species, n_wvn, n_level)  [1/m]
      p_levels         : (n_level,) [Pa]  strictly decreasing (surface→TOA)
      z_levels_hyp     : (n_level,) [m]   hypsometric from p & T
      T_levels         : (n_level,) [K]
    """
    assert w_grid is not None and len(w_grid) > 0
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, 0)
    ws.IndexSet(ws.stokes_dim, 1)
    ws.jacobianOff()
    ws.cloudboxOff()

    sp_list = list(species)
    ws.abs_speciesSet(species=sp_list)
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.ReadXsecData(basename="xsec/")

    ws.f_grid = pyarts.arts.convert.kaycm2freq(w_grid)
    ws.abs_lines_per_speciesCompact()
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesNormalization(option="SFS")
    ws.abs_lines_per_speciesTurnOffLineMixing()

    ws.propmat_clearsky_agendaAuto(T_extrapolfac=1e99)
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    ws.atm_fields_compact = atmosphere
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]
    ws.vmr_field.value = ws.vmr_field.value.value.clip(min=0.0)

    ws.cloudboxSetFullAtm()
    ws.scat_data_checked = 1
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()
    ws.sensorOff()
    ws.jacobianOff()

    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    ws.propmat_clearsky_fieldCalc()
    k_raw = np.asarray(ws.propmat_clearsky_field.value)

    shp = k_raw.shape
    nf = len(w_grid)
    nlev = int(np.asarray(ws.p_grid.value).size)
    ns = len(sp_list)

    cand_f = [i for i, s in enumerate(shp) if s == nf]
    cand_p = [i for i, s in enumerate(shp) if s == nlev]
    if len(cand_f) != 1 or len(cand_p) != 1:
        raise ValueError(
            f"Cannot identify freq/level axes from shape {shp} (nf={nf}, nlev={nlev}); "
            f"found f={cand_f}, p={cand_p}"
        )
    ax_f, ax_p = cand_f[0], cand_p[0]

    cand_s = [i for i, s in enumerate(shp) if s == ns]
    if ns > 1 and len(cand_s) == 1:
        ax_s = cand_s[0]
        k_re = np.moveaxis(k_raw, (ax_s, ax_f, ax_p), (0, 1, 2))
        k_re = np.asarray(k_re).squeeze()
        if k_re.ndim == 2:
            k_re = k_re[None, :, :]
        if k_re.shape[0] != ns or k_re.shape[1] != nf or k_re.shape[2] != nlev:
            raise ValueError(f"Unexpected reordered shape {k_re.shape}; expected (ns={ns}, nf={nf}, nlev={nlev})")
        absorption_coeff = k_re

    else:
        k_fp = np.moveaxis(k_raw, (ax_f, ax_p), (0, 1))
        k_fp = np.squeeze(k_fp)
        if k_fp.ndim != 2 or k_fp.shape != (nf, nlev):
            raise ValueError(f"Fallback expected (nf, nlev) after squeeze; got {k_fp.shape} from {shp}")
        absorption_coeff = k_fp[None, :, :]

    ws.StringSet(ws.iy_unit, "1")
    ws.disort_aux_vars = ["Layer optical thickness"]
    ws.spectral_irradiance_fieldDisort(nstreams=2, emission=1)
    tau_arts = np.asarray(ws.disort_aux.value[0].value)
    if tau_arts.shape[0] != nf:
        raise ValueError(f"tau freq dim {tau_arts.shape[0]} != len(w_grid) {nf}")

    p_levels = np.asarray(ws.p_grid.value, dtype=float).ravel()
    T_levels = np.asarray(ws.t_field.value, dtype=float)[:, 0, 0]

    R = 287.0
    g_local = 9.81
    z_levels_hyp = np.zeros_like(p_levels, dtype=float)
    for i in range(1, p_levels.size):
        p1, p2 = p_levels[i - 1], p_levels[i]
        Tbar = 0.5 * (T_levels[i - 1] + T_levels[i])
        z_levels_hyp[i] = z_levels_hyp[i - 1] + (R * Tbar / g_local) * np.log(p1 / p2)

    return tau_arts, absorption_coeff, p_levels, z_levels_hyp, T_levels


def build_reference_targets_from_lbl_fullmix(
    profile,
    nu_cm1,
    weights,
    N_LEVELS,
    mat,
    N_COLS=1,
    N_SCENARIOS=1,
):
    """
    For ONE CKDMIP column:
      - build full-mixture ARTS atmosphere with all available gases
      - run LBL on DDQ points (nu_cm1)
      - compute total k, tau, DDQ-weighted broadband fluxes and heating
      - return datasets + tensors needed downstream.
    """
    gas_names = _discover_all_gases(profile)
    if not gas_names:
        raise ValueError("No gas mole fraction fields found in profile.")

    p_hl, T_hl, vmr_dict = ensure_surface_to_toa_order(profile, gas_names)

    if p_hl.size != N_LEVELS:
        raise ValueError(f"N_LEVELS={N_LEVELS} does not match column half_levels={p_hl.size}")

    atmosphere = generate_gridded_field_from_profiles(
        p_hl,
        T_hl,
        gases={g: vmr_dict[g] for g in gas_names},
        particulates={},
        z_field=None,
    )

    tau_arts, k_species_lbl, p_levels, z_levels, T_levels = calc_lbl_rt(
        atmosphere,
        species=tuple(gas_names),
        w_grid=nu_cm1,
    )

    n_wvn = nu_cm1.size
    if k_species_lbl.shape[0] != len(gas_names):
        raise ValueError(f"Expected {len(gas_names)} species in k_species_lbl, got {k_species_lbl.shape[0]}")
    if k_species_lbl.shape[1] != n_wvn:
        raise ValueError(f"Expected n_wvn={n_wvn} in k_species_lbl, got {k_species_lbl.shape[1]}")
    if k_species_lbl.shape[2] != N_LEVELS:
        raise ValueError(f"Expected N_LEVELS={N_LEVELS} in k_species_lbl, got {k_species_lbl.shape[2]}")

    k_total_wvn_lev = np.sum(k_species_lbl, axis=0)

    tau_layer_wvn, _ = tau_from_linear_k(k_total_wvn_lev.T, z_levels, center="level")
    tau_layer_wvn = tau_layer_wvn.T

    Fu_ref, Fd_ref, Fnet_ref = broadband_flux_from_two_stream_tau(
        tau_layer_wvn,
        T_hl,
        nu_cm1,
        weights,
        to_surface_order=True,
    )

    F_ref_2D = np.tile(Fnet_ref.reshape(N_LEVELS, 1), (1, N_COLS))
    p_2D = np.tile(p_hl.reshape(N_LEVELS, 1), (1, N_COLS))

    H_ref_2D = -86400.0 * (mat @ F_ref_2D) * 9.8 / (mat @ p_2D) / 1004.0

    level = np.arange(N_LEVELS)
    column = np.arange(N_COLS)

    y_ref = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_ref_2D),
            reference_forcing_co2=(("column",), np.zeros(N_COLS)),
            reference_forcing_ch4=(("column",), np.zeros(N_COLS)),
            reference_forcing_n2o=(("column",), np.zeros(N_COLS)),
            reference_forcing_o3=(("column",), np.zeros(N_COLS)),
            reference_forcing_cfc11=(("column",), np.zeros(N_COLS)),
            reference_forcing_cfc12=(("column",), np.zeros(N_COLS)),
        ),
        coords=dict(level=level, column=column),
    )

    x_sup = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), p_2D.T),
            reference_heating=(("column", "level"), H_ref_2D.T),
        ),
        coords=dict(level=level, column=column),
    )

    return y_ref, x_sup, (gas_names, k_species_lbl, p_hl, T_hl, z_levels, vmr_dict)


# -------------------------------------------------------------------
# Caching helpers
# -------------------------------------------------------------------
def _column_cache_path(cache_dir, col_idx):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"lbl_col_{col_idx:03d}.npz")


def load_column_lbl_cache(cache_dir, col_idx, nu_cm1):
    """
    Load cached DDQ-grid LBL tensors for one column, if available.
    """
    path = _column_cache_path(cache_dir, col_idx)
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    if "nu_cm1" not in data:
        return None

    nu_saved = np.asarray(data["nu_cm1"], dtype=float)
    nu_now = np.asarray(nu_cm1, dtype=float)
    if nu_saved.shape != nu_now.shape or np.max(np.abs(nu_saved - nu_now)) > 1e-9:
        return None

    gas_names = [str(s) for s in data["gas_names"]]
    k_species_lbl = np.asarray(data["k_species_lbl"], dtype=float)
    p_hl = np.asarray(data["p_hl"], dtype=float)
    T_hl = np.asarray(data["T_hl"], dtype=float)
    z_levels = np.asarray(data["z_levels"], dtype=float)

    vmr_names = [str(s) for s in data["vmr_names"]]
    vmr_matrix = np.asarray(data["vmr_matrix"], dtype=float)
    vmr_dict = {name: vmr_matrix[i, :] for i, name in enumerate(vmr_names)}

    F_ref_col = np.asarray(data["F_ref_col"], dtype=float)
    H_ref_col = np.asarray(data["H_ref_col"], dtype=float)
    P_col = np.asarray(data["P_col"], dtype=float)

    return (
        gas_names,
        k_species_lbl,
        p_hl,
        T_hl,
        z_levels,
        vmr_dict,
        F_ref_col,
        H_ref_col,
        P_col,
    )


def save_column_lbl_cache(
    cache_dir,
    col_idx,
    nu_cm1,
    gas_names,
    k_species_lbl,
    p_hl,
    T_hl,
    z_levels,
    vmr_dict,
    F_ref_col,
    H_ref_col,
    P_col,
):
    vmr_names = np.array(list(vmr_dict.keys()))
    vmr_matrix = np.vstack([vmr_dict[name] for name in vmr_names])

    path = _column_cache_path(cache_dir, col_idx)
    np.savez_compressed(
        path,
        nu_cm1=np.asarray(nu_cm1, dtype=float),
        gas_names=np.asarray(gas_names),
        k_species_lbl=np.asarray(k_species_lbl, dtype=float),
        p_hl=np.asarray(p_hl, dtype=float),
        T_hl=np.asarray(T_hl, dtype=float),
        z_levels=np.asarray(z_levels, dtype=float),
        vmr_names=vmr_names,
        vmr_matrix=vmr_matrix,
        F_ref_col=np.asarray(F_ref_col, dtype=float),
        H_ref_col=np.asarray(H_ref_col, dtype=float),
        P_col=np.asarray(P_col, dtype=float),
    )


def compute_column_reference_and_cache(
    profile_col,
    col_idx,
    nu_cm1,
    weights,
    N_LEVELS,
    mat,
    cache_dir,
):
    """
    Run the DDQ-grid LBL reference once for this column, then cache everything needed later.
    """
    (
        y_ref_col,
        x_sup_col,
        (gas_names, k_species_lbl, p_hl, T_hl, z_levels, vmr_dict),
    ) = build_reference_targets_from_lbl_fullmix(
        profile=profile_col,
        nu_cm1=nu_cm1,
        weights=weights,
        N_LEVELS=N_LEVELS,
        mat=mat,
        N_COLS=1,
        N_SCENARIOS=1,
    )

    F_ref_col = np.asarray(y_ref_col["reference_fluxes"].values[:, 0], float)
    H_ref_col = np.asarray(x_sup_col["reference_heating"].values[0, :], float)
    P_col = np.asarray(x_sup_col["pressures"].values[0, :], float)

    save_column_lbl_cache(
        cache_dir,
        col_idx,
        nu_cm1,
        gas_names,
        k_species_lbl,
        p_hl,
        T_hl,
        z_levels,
        vmr_dict,
        F_ref_col,
        H_ref_col,
        P_col,
    )

    return (
        gas_names,
        k_species_lbl,
        p_hl,
        T_hl,
        z_levels,
        vmr_dict,
        F_ref_col,
        H_ref_col,
        P_col,
    )


def reference_dataset_from_arrays(F_net, H_ref, P_col):
    """
    Rebuild y_ref/x_sup for a single column given flux, heating, and pressure arrays.
    """
    F_arr = np.asarray(F_net, float).reshape(-1, 1)
    H_arr = np.asarray(H_ref, float).reshape(1, -1)
    P_arr = np.asarray(P_col, float).reshape(1, -1)
    n_levels = F_arr.shape[0]

    y_ref = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_arr),
            reference_forcing_co2=(("column",), np.zeros(1)),
            reference_forcing_ch4=(("column",), np.zeros(1)),
            reference_forcing_n2o=(("column",), np.zeros(1)),
            reference_forcing_o3=(("column",), np.zeros(1)),
            reference_forcing_cfc11=(("column",), np.zeros(1)),
            reference_forcing_cfc12=(("column",), np.zeros(1)),
        ),
        coords=dict(level=np.arange(n_levels), column=np.array([0])),
    )

    x_sup = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), P_arr),
            reference_heating=(("column", "level"), H_arr),
        ),
        coords=dict(level=np.arange(n_levels), column=np.array([0])),
    )

    return y_ref, x_sup


def compute_global_cost_over_all_columns_surrogate(
    profiles,
    DDQ_lw,
    kappa_arrays_dict,
    molar_mass_dict,
    target_species_list=("CO2", "O3"),
    ddq_cache_dir="lbl_ddq",
    swap_mode="replace",
):
    """
    Compute one global cost vs DDQ LBL baseline using precomputed κ_m arrays per column.
    kappa_arrays_dict expects {species: (n_wvn, n_col, n_level)} on the DDQ ν grid.

    swap_mode controls whether κ arrays are used for a true species replacement ("replace")
    """
    nu_cm1 = np.asarray(DDQ_lw.S.values, float)
    weights = np.asarray(DDQ_lw.W.values, float)

    n_cols = int(profiles.sizes["column"])
    n_levels = int(profiles.sizes["half_level"])
    mat_local = derivative_matrix(n_levels)

    F_ref_all = np.zeros((n_levels, n_cols))
    H_ref_all = np.zeros((n_cols, n_levels))
    P_all = np.zeros((n_cols, n_levels))
    F_est_all = np.zeros((n_levels, n_cols))

    iterator = range(n_cols)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Surrogate vs DDQ (arrays)", ncols=80)
    except Exception:
        pass

    for col in iterator:
        profile_col = profiles.isel(column=col)
        cached = load_column_lbl_cache(ddq_cache_dir, col, nu_cm1)
        if cached is None:
            cached = compute_column_reference_and_cache(
                profile_col=profile_col,
                col_idx=col,
                nu_cm1=nu_cm1,
                weights=weights,
                N_LEVELS=n_levels,
                mat=mat_local,
                cache_dir=ddq_cache_dir,
            )

        (
            gas_names,
            k_species_lbl,
            p_hl,
            T_hl,
            z_levels,
            vmr_dict,
            F_ref_col,
            H_ref_col,
            P_col,
        ) = cached

        F_ref_all[:, col] = F_ref_col
        H_ref_all[col, :] = H_ref_col
        P_all[col, :] = P_col

        y_hat_col = make_y_hat_with_kappa_arrays(
            target_species_list=target_species_list,
            gas_names=gas_names,
            k_species_lbl=k_species_lbl,
            p_hl=p_hl,
            T_hl=T_hl,
            z_levels=z_levels,
            nu_cm1=nu_cm1,
            weights=weights,
            kappa_arrays_dict=kappa_arrays_dict,
            molar_mass_dict=molar_mass_dict,
            vmr_dict=vmr_dict,
            col_idx=col,
            swap_mode=swap_mode,
        )
        F_est_all[:, col] = np.asarray(y_hat_col[:n_levels], float)

    level_coord = np.arange(n_levels)
    column_coord = np.arange(n_cols)

    y_ref_all = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_ref_all),
            reference_forcing_co2=(("column",), np.zeros(n_cols)),
            reference_forcing_ch4=(("column",), np.zeros(n_cols)),
            reference_forcing_n2o=(("column",), np.zeros(n_cols)),
            reference_forcing_o3=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc11=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc12=(("column",), np.zeros(n_cols)),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )

    x_sup_all = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), P_all),
            reference_heating=(("column", "level"), H_ref_all),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )

    with cost_dimension_context(n_levels, n_cols, 1):
        F_est_flat = F_est_all.T.flatten(order="F")
        zeros_forc = np.zeros(6 * n_cols)
        y_hat_all = np.concatenate([F_est_flat, zeros_forc])
        cost_val = cost_no_forcing(y_hat_all, y_ref_all, x_sup_all)
        cost_val = float(cost_val.value) if hasattr(cost_val, "value") else float(cost_val)

    return cost_val


def compute_global_cost_over_all_columns_surrogate_coeffs(
    profiles,
    DDQ_lw,
    kappa_coeffs_dict,
    molar_mass_dict,
    target_species_list=("CO2", "O3"),
    ddq_cache_dir="lbl_ddq",
    swap_mode="replace",
):
    """
    Compute one global cost vs DDQ LBL baseline using log-poly κ_m coefficients.
    kappa_coeffs_dict expects {species: (n_wvn, 6)} on the DDQ ν grid.

    swap_mode controls whether coefficients are used for a true species replacement ("replace")
    """
    nu_cm1 = np.asarray(DDQ_lw.S.values, float)
    weights = np.asarray(DDQ_lw.W.values, float)

    n_cols = int(profiles.sizes["column"])
    n_levels = int(profiles.sizes["half_level"])
    mat_local = derivative_matrix(n_levels)

    F_ref_all = np.zeros((n_levels, n_cols))
    H_ref_all = np.zeros((n_cols, n_levels))
    P_all = np.zeros((n_cols, n_levels))
    F_est_all = np.zeros((n_levels, n_cols))

    iterator = range(n_cols)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Surrogate vs DDQ (coeffs)", ncols=80)
    except Exception:
        pass

    for col in iterator:
        profile_col = profiles.isel(column=col)
        cached = load_column_lbl_cache(ddq_cache_dir, col, nu_cm1)
        if cached is None:
            cached = compute_column_reference_and_cache(
                profile_col=profile_col,
                col_idx=col,
                nu_cm1=nu_cm1,
                weights=weights,
                N_LEVELS=n_levels,
                mat=mat_local,
                cache_dir=ddq_cache_dir,
            )

        (
            gas_names,
            k_species_lbl,
            p_hl,
            T_hl,
            z_levels,
            vmr_dict,
            F_ref_col,
            H_ref_col,
            P_col,
        ) = cached

        F_ref_all[:, col] = F_ref_col
        H_ref_all[col, :] = H_ref_col
        P_all[col, :] = P_col

        y_hat_col = make_y_hat_with_species_swaps(
            target_species_list=target_species_list,
            gas_names=gas_names,
            k_species_lbl=k_species_lbl,
            p_hl=p_hl,
            T_hl=T_hl,
            z_levels=z_levels,
            nu_cm1=nu_cm1,
            weights=weights,
            kappa_coeffs_dict=kappa_coeffs_dict,
            molar_mass_dict=molar_mass_dict,
            vmr_dict=vmr_dict,
            swap_mode=swap_mode,
        )
        F_est_all[:, col] = np.asarray(y_hat_col[:n_levels], float)

    level_coord = np.arange(n_levels)
    column_coord = np.arange(n_cols)

    y_ref_all = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_ref_all),
            reference_forcing_co2=(("column",), np.zeros(n_cols)),
            reference_forcing_ch4=(("column",), np.zeros(n_cols)),
            reference_forcing_n2o=(("column",), np.zeros(n_cols)),
            reference_forcing_o3=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc11=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc12=(("column",), np.zeros(n_cols)),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )

    x_sup_all = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), P_all),
            reference_heating=(("column", "level"), H_ref_all),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )

    with cost_dimension_context(n_levels, n_cols, 1):
        F_est_flat = F_est_all.T.flatten(order="F")
        zeros_forc = np.zeros(6 * n_cols)
        y_hat_all = np.concatenate([F_est_flat, zeros_forc])
        cost_val = cost_no_forcing(y_hat_all, y_ref_all, x_sup_all)
        cost_val = float(cost_val.value) if hasattr(cost_val, "value") else float(cost_val)

    return cost_val


@contextmanager
def cost_dimension_context(n_levels, n_cols, n_scenarios=1, f_array=None):
    """
    Temporarily set the global dimensions/derivative matrix that cost_no_forcing expects.
    """
    global N_LEVELS, N_COLS, N_SCENARIOS, mat, F_ARRAY
    prev_state = (
        N_LEVELS,
        N_COLS,
        N_SCENARIOS,
        mat,
        None if F_ARRAY is None else np.array(F_ARRAY, float),
    )

    set_cost_dimensions(n_levels, n_cols, n_scenarios, f_array=f_array)
    try:
        yield
    finally:
        N_LEVELS, N_COLS, N_SCENARIOS, mat, prev_F = prev_state
        if prev_F is not None:
            F_ARRAY = prev_F


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------
def evaluate_column_costs(
    col_idx,
    profiles,
    DDQ_lw,
    nu_dense_cm1,
    kappa_coeffs_dict,
    molar_mass_dict,
    target_species_list=("CO2", "O3"),
    dense_cache_dir="lbl_dense",
    ddq_cache_dir="lbl_ddq",
):
    """
    Compare (a) DDQ full-mixture vs dense reference and (b) surrogate vs DDQ for one CKDMIP column.
    """
    nu_cm1 = np.asarray(DDQ_lw.S.values, float)
    weights = np.asarray(DDQ_lw.W.values, float)

    profile_col = profiles.isel(column=col_idx)
    n_levels = profile_col.pressure_hl.size

    y_ref_dense, x_sup_dense, _ = build_reference_targets_from_lbl_dense(
        profile=profiles,
        col=col_idx,
        nu_dense_cm1=nu_dense_cm1,
        N_LEVELS=n_levels,
        mat=derivative_matrix(n_levels),
        N_COLS=1,
        N_SCENARIOS=1,
        cache_dir=dense_cache_dir,
    )
    dense_flux = np.asarray(y_ref_dense.reference_fluxes.values[:, 0], float)
    dense_heating = np.asarray(x_sup_dense.reference_heating.values[0, :], float)
    dense_pressures = np.asarray(x_sup_dense.pressures.values[0, :], float)

    cached = load_column_lbl_cache(ddq_cache_dir, col_idx, nu_cm1)
    if cached is None:
        cached = compute_column_reference_and_cache(
            profile_col=profile_col,
            col_idx=col_idx,
            nu_cm1=nu_cm1,
            weights=weights,
            N_LEVELS=n_levels,
            mat=derivative_matrix(n_levels),
            cache_dir=ddq_cache_dir,
        )

    (
        gas_names,
        k_species_lbl,
        p_hl,
        T_hl,
        z_levels,
        vmr_dict,
        F_ref_col,
        H_ref_col,
        P_col,
    ) = cached

    y_ref_ddq, x_sup_ddq = reference_dataset_from_arrays(F_ref_col, H_ref_col, P_col)

    y_hat_ddq = make_y_hat_with_species_swaps(
        target_species_list=[],
        gas_names=gas_names,
        k_species_lbl=k_species_lbl,
        p_hl=p_hl,
        T_hl=T_hl,
        z_levels=z_levels,
        nu_cm1=nu_cm1,
        weights=weights,
        kappa_coeffs_dict=kappa_coeffs_dict,
        molar_mass_dict=molar_mass_dict,
        vmr_dict=vmr_dict,
    )

    y_hat_surrogate = make_y_hat_with_species_swaps(
        target_species_list=target_species_list,
        gas_names=gas_names,
        k_species_lbl=k_species_lbl,
        p_hl=p_hl,
        T_hl=T_hl,
        z_levels=z_levels,
        nu_cm1=nu_cm1,
        weights=weights,
        kappa_coeffs_dict=kappa_coeffs_dict,
        molar_mass_dict=molar_mass_dict,
        vmr_dict=vmr_dict,
    )

    with cost_dimension_context(n_levels, 1, 1):
        cost_ddq_vs_dense = float(cost_no_forcing(y_hat_ddq, y_ref_dense, x_sup_dense).value)
        cost_surrogate_vs_ddq = float(cost_no_forcing(y_hat_surrogate, y_ref_ddq, x_sup_ddq).value)
        cost_surrogate_vs_dense = float(cost_no_forcing(y_hat_surrogate, y_ref_dense, x_sup_dense).value)

    return {
        "cost_ddq_vs_dense": cost_ddq_vs_dense,
        "cost_surrogate_vs_ddq": cost_surrogate_vs_ddq,
        "cost_surrogate_vs_dense": cost_surrogate_vs_dense,
        "dense_flux": dense_flux,
        "dense_heating": dense_heating,
        "dense_pressures": dense_pressures,
        "ddq_flux": F_ref_col,
        "ddq_heating": H_ref_col,
        "ddq_pressures": P_col,
        "ddq_est_flux": y_hat_ddq[:n_levels],
        "surrogate_flux": y_hat_surrogate[:n_levels],
    }


def compute_global_costs_ddq_vs_dense(
    profiles,
    DDQ_lw,
    nu_dense_cm1,
    kappa_coeffs_dict,
    molar_mass_dict,
    target_species_list=("CO2", "O3"),
    dense_cache_dir="lbl_dense",
    ddq_cache_dir="lbl_ddq",
    kappa_arrays_dict=None,
):
    """
    Compute both requested global costs:
      1) DDQ full-mixture vs dense LBL baseline.
      2) Surrogate κm vs DDQ LBL baseline.
    If kappa_arrays_dict is provided, use per-column κ_m arrays (n_wvn, n_col, n_level)
    instead of log-poly coefficients.
    """
    nu_cm1 = np.asarray(DDQ_lw.S.values, float)
    weights = np.asarray(DDQ_lw.W.values, float)
    if nu_cm1.ndim != 1 or weights.ndim != 1 or nu_cm1.size != weights.size:
        raise ValueError("DDQ S and W must be one dimensional and aligned.")

    n_cols = int(profiles.sizes["column"])
    n_levels = int(profiles.sizes["half_level"])
    mat_local = derivative_matrix(n_levels)

    F_ref_dense = np.zeros((n_levels, n_cols))
    H_ref_dense = np.zeros((n_cols, n_levels))
    P_ref_dense = np.zeros((n_cols, n_levels))

    F_ref_ddq = np.zeros((n_levels, n_cols))
    H_ref_ddq = np.zeros((n_cols, n_levels))
    P_ref_ddq = np.zeros((n_cols, n_levels))

    F_ddq_est = np.zeros((n_levels, n_cols))
    F_surrogate = np.zeros((n_levels, n_cols))

    iterator = range(n_cols)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Global DDQ vs dense", ncols=80)
    except Exception:
        pass

    for col in iterator:
        profile_col = profiles.isel(column=col)

        y_ref_dense_col, x_sup_dense_col, _ = build_reference_targets_from_lbl_dense(
            profile=profiles,
            col=col,
            nu_dense_cm1=nu_dense_cm1,
            N_LEVELS=n_levels,
            mat=mat_local,
            N_COLS=1,
            N_SCENARIOS=1,
            cache_dir=dense_cache_dir,
        )
        dense_flux = np.asarray(y_ref_dense_col.reference_fluxes.values[:, 0], float)
        dense_heating = np.asarray(x_sup_dense_col.reference_heating.values[0, :], float)
        dense_pressures = np.asarray(x_sup_dense_col.pressures.values[0, :], float)

        F_ref_dense[:, col] = dense_flux
        H_ref_dense[col, :] = dense_heating
        P_ref_dense[col, :] = dense_pressures

        cached = load_column_lbl_cache(ddq_cache_dir, col, nu_cm1)
        if cached is None:
            cached = compute_column_reference_and_cache(
                profile_col=profile_col,
                col_idx=col,
                nu_cm1=nu_cm1,
                weights=weights,
                N_LEVELS=n_levels,
                mat=mat_local,
                cache_dir=ddq_cache_dir,
            )

        (
            gas_names,
            k_species_lbl,
            p_hl,
            T_hl,
            z_levels,
            vmr_dict,
            F_ref_col,
            H_ref_col,
            P_col,
        ) = cached

        F_ref_ddq[:, col] = F_ref_col
        H_ref_ddq[col, :] = H_ref_col
        P_ref_ddq[col, :] = P_col

        y_hat_ddq = make_y_hat_with_species_swaps(
            target_species_list=[],
            gas_names=gas_names,
            k_species_lbl=k_species_lbl,
            p_hl=p_hl,
            T_hl=T_hl,
            z_levels=z_levels,
            nu_cm1=nu_cm1,
            weights=weights,
            kappa_coeffs_dict=kappa_coeffs_dict,
            molar_mass_dict=molar_mass_dict,
            vmr_dict=vmr_dict,
        )
        if kappa_arrays_dict is not None:
            y_hat_surrogate = make_y_hat_with_kappa_arrays(
                target_species_list=target_species_list,
                gas_names=gas_names,
                k_species_lbl=k_species_lbl,
                p_hl=p_hl,
                T_hl=T_hl,
                z_levels=z_levels,
                nu_cm1=nu_cm1,
                weights=weights,
                kappa_arrays_dict=kappa_arrays_dict,
                molar_mass_dict=molar_mass_dict,
                vmr_dict=vmr_dict,
                col_idx=col,
            )
        else:
            y_hat_surrogate = make_y_hat_with_species_swaps(
                target_species_list=target_species_list,
                gas_names=gas_names,
                k_species_lbl=k_species_lbl,
                p_hl=p_hl,
                T_hl=T_hl,
                z_levels=z_levels,
                nu_cm1=nu_cm1,
                weights=weights,
                kappa_coeffs_dict=kappa_coeffs_dict,
                molar_mass_dict=molar_mass_dict,
                vmr_dict=vmr_dict,
            )

        F_ddq_est[:, col] = y_hat_ddq[:n_levels]
        F_surrogate[:, col] = y_hat_surrogate[:n_levels]

    level_coord = np.arange(n_levels)
    column_coord = np.arange(n_cols)

    y_ref_dense_all = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_ref_dense),
            reference_forcing_co2=(("column",), np.zeros(n_cols)),
            reference_forcing_ch4=(("column",), np.zeros(n_cols)),
            reference_forcing_n2o=(("column",), np.zeros(n_cols)),
            reference_forcing_o3=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc11=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc12=(("column",), np.zeros(n_cols)),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )
    x_sup_dense_all = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), P_ref_dense),
            reference_heating=(("column", "level"), H_ref_dense),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )

    y_ref_ddq_all = xr.Dataset(
        data_vars=dict(
            reference_fluxes=(("level", "column"), F_ref_ddq),
            reference_forcing_co2=(("column",), np.zeros(n_cols)),
            reference_forcing_ch4=(("column",), np.zeros(n_cols)),
            reference_forcing_n2o=(("column",), np.zeros(n_cols)),
            reference_forcing_o3=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc11=(("column",), np.zeros(n_cols)),
            reference_forcing_cfc12=(("column",), np.zeros(n_cols)),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )
    x_sup_ddq_all = xr.Dataset(
        data_vars=dict(
            pressures=(("column", "level"), P_ref_ddq),
            reference_heating=(("column", "level"), H_ref_ddq),
        ),
        coords=dict(level=level_coord, column=column_coord),
    )

    with cost_dimension_context(n_levels, n_cols, 1):
        zeros_forc = np.zeros(6 * n_cols)
        y_hat_ddq_all = np.concatenate([F_ddq_est.T.flatten(order="F"), zeros_forc])
        y_hat_surrogate_all = np.concatenate([F_surrogate.T.flatten(order="F"), zeros_forc])

        cost_ddq_vs_dense = float(cost_no_forcing(y_hat_ddq_all, y_ref_dense_all, x_sup_dense_all).value)
        cost_surrogate_vs_ddq = float(cost_no_forcing(y_hat_surrogate_all, y_ref_ddq_all, x_sup_ddq_all).value)
        cost_surrogate_vs_dense = float(cost_no_forcing(y_hat_surrogate_all, y_ref_dense_all, x_sup_dense_all).value)

    print(
        f"Global cost (DDQ baseline vs dense): {cost_ddq_vs_dense:.6f}  |  "
        f"Global cost (surrogate vs DDQ): {cost_surrogate_vs_ddq:.6f}"
    )

    return dict(
        cost_ddq_vs_dense=cost_ddq_vs_dense,
        cost_surrogate_vs_ddq=cost_surrogate_vs_ddq,
        cost_surrogate_vs_dense=cost_surrogate_vs_dense,
        F_ref_dense=F_ref_dense,
        F_ref_ddq=F_ref_ddq,
        F_ddq_est=F_ddq_est,
        F_surrogate=F_surrogate,
        H_ref_dense=H_ref_dense,
        H_ref_ddq=H_ref_ddq,
        pressures_dense=P_ref_dense,
        pressures_ddq=P_ref_ddq,
    )


# -------------------------------------------------------------------
# Diagnostics + plotting helpers (per-column error summaries)
# -------------------------------------------------------------------
def heating_from_flux_and_pressure(F_net_level_col, pressures_col_level, mat_local=None, g_local=9.81, cp_air=1004.0, seconds_per_day=86400.0):
    """
    Recompute heating rate from broadband net flux and pressure using the same algebra as the cost function.

    Inputs:
      F_net_level_col     : (n_level, n_col)
      pressures_col_level : (n_col, n_level)
    Returns:
      H_level_col : (n_level, n_col) in K/day
    """
    F = np.asarray(F_net_level_col, float)
    P_cl = np.asarray(pressures_col_level, float)
    if F.ndim != 2 or P_cl.ndim != 2:
        raise ValueError("F_net_level_col and pressures_col_level must be 2-D arrays.")
    n_levels, n_cols = F.shape
    if P_cl.shape != (n_cols, n_levels):
        raise ValueError(f"pressures must be (n_col,n_level) = ({n_cols},{n_levels}); got {P_cl.shape}")

    mat_use = derivative_matrix(n_levels) if mat_local is None else np.asarray(mat_local, float)
    P_lc = P_cl.T
    H = -seconds_per_day * (mat_use @ F) * g_local / (mat_use @ P_lc) / cp_air
    return np.asarray(H, float)


def _iqr_outlier_mask(values):
    vals = np.asarray(values, float)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return (vals < lo) | (vals > hi)


def summarize_column_rms_flux_heating(
    F_hat_level_col,
    F_ref_level_col,
    pressures_col_level,
    H_ref_col_level=None,
    label="",
    scenario="",
    mat_local=None,
):
    """
    Build per-column RMS summaries for plotting.
    """
    F_hat = np.asarray(F_hat_level_col, float)
    F_ref = np.asarray(F_ref_level_col, float)
    if F_hat.shape != F_ref.shape:
        raise ValueError(f"F_hat and F_ref must have same shape; got {F_hat.shape} vs {F_ref.shape}")
    n_levels, n_cols = F_hat.shape

    # Flux RMS per column
    col_rms_flux = np.sqrt(np.mean((F_hat - F_ref) ** 2, axis=0))

    # Heating RMS per column
    H_hat_lc = heating_from_flux_and_pressure(F_hat, pressures_col_level, mat_local=mat_local)
    if H_ref_col_level is None:
        H_ref_lc = heating_from_flux_and_pressure(F_ref, pressures_col_level, mat_local=mat_local)
    else:
        H_ref = np.asarray(H_ref_col_level, float)
        if H_ref.shape != (n_cols, n_levels):
            raise ValueError(f"H_ref_col_level must be (n_col,n_level)=({n_cols},{n_levels}); got {H_ref.shape}")
        H_ref_lc = H_ref.T
    col_rms_heat = np.sqrt(np.mean((H_hat_lc - H_ref_lc) ** 2, axis=0))

    return {
        "label": str(label),
        "scenario": str(scenario),
        "cols": int(n_cols),
        "levels": int(n_levels),
        "col_rms_flux": col_rms_flux,
        "col_rms_heat": col_rms_heat,
    }


def plot_column_rms_histograms_labeled(
    results,
    bins=20,
    label_sparse_bins=True,
    sparse_threshold=2,
    label_iqr_outliers=True,
    print_tables=True,
):
    """
    Histogram diagnostics for per-column RMS (flux + heating).
    Labels columns that are IQR outliers and/or land in sparse histogram bins.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    n_cols = int(results["cols"])
    col_ids = np.arange(n_cols)
    rms_F = np.asarray(results["col_rms_flux"], float)
    rms_H = np.asarray(results["col_rms_heat"], float)

    def _plot_one(ax, vals, title, xlabel):
        counts, edges = np.histogram(vals, bins=bins)
        ax.hist(vals, bins=edges, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")

        sparse_bins = set(np.where(counts <= sparse_threshold)[0]) if label_sparse_bins else set()
        bin_idx = np.digitize(vals, edges, right=True) - 1
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

        by_bin = {}
        for c, b in zip(col_ids, bin_idx):
            by_bin.setdefault(int(b), []).append(int(c))

        iqr_mask = _iqr_outlier_mask(vals) if label_iqr_outliers else np.zeros_like(vals, dtype=bool)
        iqr_cols = set(col_ids[iqr_mask].tolist())

        cols_to_label = set()
        for b in sparse_bins:
            cols_to_label.update(by_bin.get(int(b), []))
        cols_to_label.update(iqr_cols)

        ymax = counts.max() if len(counts) else 0
        pad = 0.05 * (ymax if ymax > 0 else 1.0)

        for b in sorted(set(bin_idx[c] for c in cols_to_label)):
            cols_here = [c for c in cols_to_label if int(bin_idx[c]) == int(b)]
            if not cols_here:
                continue
            x_center = 0.5 * (edges[int(b)] + edges[int(b) + 1])
            y_bar = counts[int(b)]
            cols_here_sorted = sorted(cols_here, key=lambda c: vals[c])
            n_labels = len(cols_here_sorted)
            bin_width = edges[int(b) + 1] - edges[int(b)]

            if n_labels == 1:
                x_positions = [x_center]
            else:
                spread = bin_width * 0.6
                x_positions = np.linspace(x_center - spread / 2.0, x_center + spread / 2.0, n_labels)

            y_pos = y_bar + pad
            for j, c in enumerate(cols_here_sorted):
                ax.text(
                    x_positions[j],
                    y_pos,
                    f"col {c} ({vals[c]:.3g})",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                )

        if label_iqr_outliers:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            for v in (q1, q3, lo, hi):
                ax.axvline(v, linestyle="--", linewidth=1)

        table = None
        if print_tables and cols_to_label:
            table = (
                pd.DataFrame(
                    {
                        "column": sorted(cols_to_label),
                        "value": [vals[c] for c in sorted(cols_to_label)],
                        "bin_index": [int(bin_idx[c]) for c in sorted(cols_to_label)],
                    }
                )
                .sort_values("value")
                .reset_index(drop=True)
            )
            print(f"\nLabeled columns for {title} ({results.get('label','')} | {results.get('scenario','')}):")
            print(table.to_string(index=False))
        return table

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    title_prefix = f"{results.get('label','')}".strip()
    if results.get("scenario"):
        title_prefix = f"{title_prefix} ({results['scenario']})".strip()

    t1 = f"{title_prefix} flux RMS" if title_prefix else "Flux RMS"
    t2 = f"{title_prefix} heating RMS" if title_prefix else "Heating RMS"

    table_F = _plot_one(ax1, rms_F, t1, "RMS net flux error [W m$^{-2}$]")
    table_H = _plot_one(ax2, rms_H, t2, "RMS heating error [K day$^{-1}$]")

    plt.tight_layout()
    plt.show()
    return {"flux_table": table_F, "heating_table": table_H}


def plot_flux_vs_heating_rms_scatter(results, label_iqr_outliers=True):
    """
    Scatter plot: flux RMS vs heating RMS across columns.
    """
    import matplotlib.pyplot as plt

    n_cols = int(results["cols"])
    col_ids = np.arange(n_cols)
    rms_F = np.asarray(results["col_rms_flux"], float)
    rms_H = np.asarray(results["col_rms_heat"], float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rms_F, rms_H, s=20, alpha=0.8)
    ax.set_xlabel("RMS net flux error [W m$^{-2}$]")
    ax.set_ylabel("RMS heating error [K day$^{-1}$]")
    title = results.get("label", "").strip()
    if results.get("scenario"):
        title = f"{title} ({results['scenario']})".strip()
    ax.set_title(title or "Flux vs heating RMS")
    ax.grid(alpha=0.3)

    if label_iqr_outliers:
        out_F = set(col_ids[_iqr_outlier_mask(rms_F)].tolist())
        out_H = set(col_ids[_iqr_outlier_mask(rms_H)].tolist())
        out = sorted(out_F.union(out_H))
        for c in out:
            ax.text(rms_F[c], rms_H[c], f"{c}", fontsize=8, ha="left", va="bottom")

    plt.tight_layout()
    plt.show()
