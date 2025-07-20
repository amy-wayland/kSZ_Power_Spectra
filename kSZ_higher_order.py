import numpy as np
#import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

import scipy.integrate as si

# Define a replacement function for simpson that uses trapz internally
def simpson_replacement(y, *args, **kwargs):
    # Extract optional x array argument for integration
    x = kwargs.pop('x', None)
    if x is None and args:
        x = args[0]
    if x is None:
        return np.trapz(y)
    else:
        return np.trapz(y, x)

# Monkey patch scipy.integrate.simpson before importing or using pyccl
si.simpson = simpson_replacement

import pyccl as ccl

#%%
# Perturbation theory kernels

#%%

def P_L(k_vec, P_L_interp):
    k_norm = np.clip(np.linalg.norm(k_vec), 1e-4, 50)
    return max(P_L_interp(k_norm), 0.0)


def F_2(k_1_vec, k_2_vec):
    
    k_1_norm = np.linalg.norm(k_1_vec)
    k_2_norm = np.linalg.norm(k_2_vec)
    dot_prod = np.dot(k_1_vec, k_2_vec)
    mu_12 = dot_prod / (k_1_norm * k_2_norm)
    
    if k_1_norm < 1e-12 or k_2_norm < 1e-12:
        return 0.0
    
    return 5/7 + (mu_12 / 2) * ((k_1_norm / k_2_norm) + (k_2_norm / k_1_norm)) + (2/7) * mu_12**2


def G_2(k_1_vec, k_2_vec):
    
    k_1_norm = np.linalg.norm(k_1_vec)
    k_2_norm = np.linalg.norm(k_2_vec)
    dot_prod = np.dot(k_1_vec, k_2_vec)
    mu_12 = dot_prod / (k_1_norm * k_2_norm)
    
    if k_1_norm < 1e-12 or k_2_norm < 1e-12:
        return 0.0
    
    return 3/7 + (mu_12/2) * ((k_1_norm / k_2_norm) + (k_2_norm / k_1_norm)) + (4/7) * mu_12**2


def Q(k_1_vec, k_2_vec, k_3_vec):
    
    k_123_vec = k_1_vec + k_2_vec + k_3_vec
    k_23_vec = k_2_vec + k_3_vec
    k_1_norm = np.linalg.norm(k_1_vec)
    k_123_norm = np.linalg.norm(k_123_vec)
    k_23_norm = np.linalg.norm(k_23_vec)
    
    if k_1_norm < 1e-12 or k_23_norm < 1e-12:
        return 0.0
    
    prefactor_1 = (7 * np.dot(k_123_vec, k_1_vec)) / k_1_norm**2
    term_1 = prefactor_1 * F_2(k_2_vec, k_3_vec)
    prefactor_2 = (7 * np.dot(k_123_vec, k_23_vec)) / k_23_norm**2 + (2 * k_123_norm**2 * np.dot(k_23_vec, k_1_vec)) / (k_23_norm**2 * k_1_norm**2)
    term_2 = prefactor_2 * G_2(k_2_vec, k_3_vec)
    
    return term_1 + term_2


def F_3(k_1_vec, k_2_vec, k_3_vec):
    return (1/54) * (Q(k_1_vec, k_2_vec, k_3_vec) + Q(k_2_vec, k_3_vec, k_1_vec) + Q(k_3_vec, k_1_vec, k_2_vec))
    

def T_1122(k, kp, kpp, P_L_interp):
    
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]
    
    total = 0.0
    indices = [
        (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2),
        (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 3, 0), (2, 3, 1)]

    for i, j, k in indices:
        q_vec = ks[i] + ks[k]
        F2_1 = F_2(q_vec, -ks[i])
        F2_2 = F_2(q_vec, ks[j])
        P1 = P_L(q_vec, P_L_interp)
        P2 = P_L(ks[i], P_L_interp)
        P3 = P_L(ks[j], P_L_interp)
        total += F2_1 * F2_2 * P1 * P2 * P3

    return 4 * total


def T_1113(k, kp, kpp, P_L_interp):
    
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]

    total = 0.0
    indices = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]

    for i, j, l in indices:
        F3_val = F_3(ks[i], ks[j], ks[l])
        P1 = P_L(ks[i], P_L_interp)
        P2 = P_L(ks[j], P_L_interp)
        P3 = P_L(ks[l], P_L_interp)
        total += F3_val * P1 * P2 * P3

    return 6 * total


def T_tree_level(k, kp, kpp, P_L_interp):
    return T_1122(k, kp, kpp, P_L_interp) + T_1113(k, kp, kpp, P_L_interp)


def B_tree_level(k_1_vec, k_2_vec, k_3_vec, P_L_interp):
    B_13 = 2 * F_2(k_1_vec, k_2_vec) * P_L(k_1_vec, P_L_interp) * P_L(k_2_vec, P_L_interp)
    B_23 = 2 * F_2(k_2_vec, k_3_vec) * P_L(k_2_vec, P_L_interp) * P_L(k_3_vec, P_L_interp)
    B_31 = 2 * F_2(k_3_vec, k_1_vec) * P_L(k_3_vec, P_L_interp) * P_L(k_1_vec, P_L_interp)
    return B_13 + B_23 + B_31
    

#%%
# Precompute halo properties

#%%

def precompute_nM(cosmo, M_vals, a):
    '''
    Precompute n(M) at given mass values
    '''
    nM_vals = nM(cosmo, M_vals, a)
    return nM_vals


def precompute_bM(cosmo, M_vals, a):
    '''
    Precompute b(M) at given mass values
    '''
    bM_vals = bM(cosmo, M_vals, a)
    return bM_vals


#%%
# Functions to calculate the halo model contributions

#%%

def trispectrum_1h(k, kp, kpp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, P_L_interp):
    '''
    Calculates the 1-halo contribution to the trispectrum

    '''
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]

    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)

    u1 = interp_pg.ev(np.log10(k_norms[0]), log10M)
    u2 = interp_pe.ev(np.log10(k_norms[1]), log10M)
    u3 = interp_pM.ev(np.log10(k_norms[2]), log10M)
    u4 = interp_pM.ev(np.log10(k_norms[3]), log10M)

    integrand = nM_vals * bM_vals * u1 * u2 * u3 * u4
    integral = np.trapz(integrand, log10M)
    result = integral
    #print(f"trispectrum_1h = {result:.3e}")
    return result
    

def trispectrum_4h(k, kp, kpp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, P_L_interp):
    '''
    Calculates the 4-halo contribution to the trispectrum

    '''
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]
    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)

    # Evaluate each profile over (k, M) grid
    u1_vals = interp_pg.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pe.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pM.ev(np.log10(k_norms[2]), log10M)
    u4_vals = interp_pM.ev(np.log10(k_norms[3]), log10M)

    # Compute separate integrals for each u_i
    integrand1 = nM_vals * bM_vals * u1_vals
    integrand2 = nM_vals * bM_vals * u2_vals
    integrand3 = nM_vals * bM_vals * u3_vals
    integrand4 = nM_vals * bM_vals * u4_vals

    I1 = np.trapz(integrand1, log10M)
    I2 = np.trapz(integrand2, log10M)
    I3 = np.trapz(integrand3, log10M)
    I4 = np.trapz(integrand4, log10M)

    T_tree = T_tree_level(k, kp, kpp, P_L_interp)
    result = T_tree * I1 * I2 * I3 * I4
    #print(f"trispectrum_4h = {result:.3e}")
    return result


def bispectrum_1h(k, kp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pa, interp_pM, P_L_interp):
    '''
    Calculates the 1-halo contribution to the bispectrum

    '''
    k1 = k
    k2 = -kp
    k3 = - (k - kp)
    ks = [k1, k2, k3]
    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)

    # Evaluate each profile over (k, M) grid
    u1_vals = interp_pM.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pM.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pa.ev(np.log10(k_norms[2]), log10M)

    integrand = nM_vals * bM_vals * u1_vals * u2_vals * u3_vals
    integral = np.trapz(integrand, log10M)
    result = integral
    #print(f"bispectrum_1h = {result:.3e}")
    return result


def bispectrum_3h(k, kp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pa, interp_pM, P_L_interp):
    '''
    Calculates the 3-halo contribution to the bispectrum

    '''
    
    k1 = k
    k2 = -kp
    k3 = - (k - kp)
    ks = [k1, k2, k3]
    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)

    # Evaluate each profile over (k, M) grid
    u1_vals = interp_pM.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pM.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pa.ev(np.log10(k_norms[2]), log10M)

    # Compute separate integrals for each u_i
    integrand1 = nM_vals * bM_vals * u1_vals
    integrand2 = nM_vals * bM_vals * u2_vals
    integrand3 = nM_vals * bM_vals * u3_vals

    I1 = np.trapz(integrand1, log10M)
    I2 = np.trapz(integrand2, log10M)
    I3 = np.trapz(integrand3, log10M)

    B_tree = B_tree_level(k1, k2, k3, P_L_interp)
    result = B_tree * I1 * I2 * I3
    #print(f"bispectrum_3h = {result:.3e}")
    return result


#%%
# Functions to compute the integrals

#%%

def P_T_perp(k_vec, cosmo, a, aHf, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_func, P_L_interp, n_k=24, n_mu=24, n_phi=28):

    k_mag = np.linalg.norm(k_vec)
    if k_mag < 1e-5:
        return 0.0
    
    # Integration grids
    lk_min, lk_max = -3, 1
    lkps = np.linspace(lk_min, lk_max, n_k)
    lkpps = np.linspace(lk_min, lk_max, n_k)
    kps = 10**lkps
    kpps = 10**lkpps
    mu_primes = np.linspace(-0.99, 0.99, n_mu)
    mupp_primes = np.linspace(-0.99, 0.99, n_mu)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)

    dlk = (lk_max - lk_min) / (n_k - 1)
    dmu = 2 / n_mu
    dphi = 2 * np.pi / n_phi

    prefactor = aHf**2 / (2 * np.pi)**5
    result = 0.0

    for i, kp in enumerate(kps):
        for j, kpp in enumerate(kpps):
            for mu1 in mu_primes:
                sin_theta1 = np.sqrt(1 - mu1**2)

                # Fixed k' vector in the x-z plane (due to azimuthal symmetry)
                kp_vec = kp * np.array([sin_theta1, 0.0, mu1])

                for mu2 in mupp_primes:
                    sin_theta2 = np.sqrt(1 - mu2**2)

                    # Vectorised k'' vectors over phi
                    kpp_vecs = kpp * np.stack([sin_theta2 * cos_phis, sin_theta2 * sin_phis, mu2 * np.ones_like(phis)], axis=1)

                    # Evaluate trispectrum for each k''
                    T_vals = np.array([trispectrum_func(k_vec, kp_vec, kpp_vec, cosmo, a,M_vals, nM_vals, bM_vals,interp_pg, interp_pe, interp_pM, P_L_interp) for kpp_vec in kpp_vecs])
                    
                    # Approximate the 5D integral as a weighted 5D sum of the integrand times the volume elements
                    block_sum = np.sum(kp * kpp * sin_theta1 * sin_theta2 * cos_phis * T_vals * kp * kpp * dlk * dlk * dmu * dmu * dphi)
                    result += block_sum

    final_result = prefactor * result
    return final_result


def P_T_par(k_vec, cosmo, a, aHf, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_func, P_L_interp, n_k=24, n_mu=24, n_phi=28):

    k_mag = (np.dot(k_vec, k_vec))**(1/2)
    if k_mag < 1e-5:
        return 0.0

    # Integration grids
    lk_min, lk_max = -3, 1
    lkps = np.linspace(lk_min, lk_max, n_k)
    lkpps = np.linspace(lk_min, lk_max, n_k)
    kps = 10**lkps
    kpps = 10**lkpps
    mu_primes = np.linspace(-0.99, 0.99, n_mu)
    mupp_primes = np.linspace(-0.99, 0.99, n_mu)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    
    sin_theta1s = np.sqrt(1 - mu_primes**2)
    sin_theta2s = np.sqrt(1 - mupp_primes**2)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)

    dlk = (lk_max - lk_min) / (n_k - 1)
    dmu = 2 / n_mu
    dphi = 2 * np.pi / n_phi

    prefactor = aHf**2 / (2 * np.pi)**5
    result = 0.0

    for i, kp in enumerate(kps):
        for j, kpp in enumerate(kpps):
            for m, mu1 in enumerate(mu_primes):
                sin_theta1 = sin_theta1s[m]

                # Fixed k' vector
                kp_vec = kp * np.array([sin_theta1, 0.0, mu1])

                for n, mu2 in enumerate(mupp_primes):
                    sin_theta2 = sin_theta2s[n]

                    # Vectorised k'' vectors
                    kpp_vecs = kpp * np.stack([sin_theta2 * cos_phis, sin_theta2 * sin_phis, mu2 * np.ones_like(phis)], axis=1)

                    # Evaluate trispectrum for each phi
                    T_vals = np.array([trispectrum_func(k_vec, kp_vec, kpp_vec, cosmo, a, M_vals, nM_vals, bM_vals,interp_pg, interp_pe, interp_pM, P_L_interp) for kpp_vec in kpp_vecs])

                    block_sum = np.sum(kp * kpp * mu1 * mu2 * T_vals * kp * kpp * dlk * dlk * dmu * dmu * dphi)
                    result += block_sum

    final_result = prefactor * result / k_mag**2
    return final_result


def P_B_par(k_vec, cosmo, a, aHf, M_vals, nM_vals, bM_vals, interp_pa, interp_pM, bispectrum_func, P_L_interp, n_k=20, n_mu=20, n_phi=24):
    
    k_mag = np.linalg.norm(k_vec)
    if k_mag < 1e-5:
        return 0.0

    # Integration grids
    lk_min, lk_max = -3, 1
    lkps = np.linspace(lk_min, lk_max, n_k)
    kps = 10**lkps
    mu_primes = np.linspace(-0.99, 0.99, n_mu)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    
    sin_thetas = np.sqrt(1 - mu_primes**2)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)

    dlk = (lk_max - lk_min) / (n_k - 1)
    dmu = 2 / n_mu
    dphi = 2 * np.pi / n_phi

    prefactor = aHf**2 / ((2 * np.pi)**3 * k_mag**2)
    result = 0.0

    for i, kp in enumerate(kps):
        for m, mu in enumerate(mu_primes):
            sin_theta = sin_thetas[m]

            # k' vectors over phi
            kp_vecs = kp * np.stack([sin_theta * cos_phis, sin_theta * sin_phis, mu * np.ones_like(phis)], axis=1)

            # Evaluate bispectrum for each phi
            B_vals = np.array([
                bispectrum_func(k_vec, kp_vec, cosmo, a, M_vals, nM_vals, bM_vals, interp_pa, interp_pM, P_L_interp)
                for kp_vec in kp_vecs])

            block_sum = np.sum(k_mag * kp * mu * B_vals * kp * dlk * dmu * dphi)
            result += block_sum

    final_result = prefactor * result
    return final_result


#%%
# Cosmology

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

k_vals = np.logspace(-3, 1, 128)
k_prime_vals = np.logspace(-3, 1, 128)
lk_arr = np.log(k_vals)
a_arr = np.linspace(0.1, 1, 32)

log10M = np.linspace(11, 15, 128)
M = 10**log10M

z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = a * H * f

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

#%%
# Overdensities

#%%

# Halo mass definition
hmd_200m = ccl.halos.MassDef200m

# Concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

# Mass function
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)

# Halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

# Matter overdensity 
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

# Galaxy overdensity
pG = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=12.89, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1, bg_0=2.04)

# Halo model integral calculator
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=10., nM=32)

# Gas density profile
profile_parameters = {"lMc": 14.0, "beta": 0.6, "eta_b": 0.5, "A_star": 0.03}
pGas = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
pGas.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=300, plaw_fourier=-2.0)

# Precompute halo properties
nM_vals = precompute_nM(cosmo, M, a)
bM_vals = precompute_bM(cosmo, M, a)

#%%
# Precompute and interpolate Fourier transforms

#%%

# Define grids for M and k
log10M_grid = np.linspace(11, 15, 128)
M_grid = 10**log10M_grid
k_grid = np.logspace(-3, 2, 128)
log10k_grid = np.log10(k_grid)

# Arrays for Fourier transforms
u_pM_grid = np.zeros((len(k_grid), len(M_grid)))
u_pg_grid = np.zeros((len(k_grid), len(M_grid)))
u_pe_grid = np.zeros((len(k_grid), len(M_grid)))

# Fill the grids
for i, k_val in enumerate(k_grid):
    for j, M_val in enumerate(M_grid):
        u_pM_grid[i, j] = pM.fourier(cosmo, k_val, M_val, a) / M_val
        u_pg_grid[i, j] = pG.fourier(cosmo, k_val, M_val, a) / pG.get_normalization(cosmo, a, hmc=hmc)
        u_pe_grid[i, j] = pGas.fourier(cosmo, k_val, M_val, a)

interp_pg = RectBivariateSpline(log10k_grid, log10M_grid, u_pg_grid)
interp_pe = RectBivariateSpline(log10k_grid, log10M_grid, u_pe_grid)
interp_pM = RectBivariateSpline(log10k_grid, log10M_grid, u_pM_grid)

k_dense = np.logspace(-3, 1, 1000)
P_dense = ccl.linear_matter_power(cosmo, k_dense, a)
P_L_interp = interp1d(k_dense, P_dense, bounds_error=False, fill_value=0.0)

#%%
# Calculations

#%%

# 4-halo contribution to the trispectrum for the transverse mode
def compute_P_T_4h_perp(k):
    return P_T_perp(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_4h, P_L_interp)

# 1-halo contribution to the trispectrum for the longitudinal mode
def compute_P_T_1h_par(k):
    return P_T_par(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_1h, P_L_interp)

# 4-halo contribution to the trispectrum for the longitudinal mode
def compute_P_T_4h_par(k):
    return P_T_par(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_4h, P_L_interp)

# 1-halo contribution to the bispectrum for the longitudinal mode for mme
def compute_P_B_mme_1h(k):
    return P_B_par(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pe, interp_pM, bispectrum_1h, P_L_interp)

# 1-halo contribution to the bispectrum for the longitudinal mode for mmg
def compute_P_B_mmg_1h(k):
    return P_B_par(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pg, interp_pM, bispectrum_1h, P_L_interp)

# 3-halo contribution to the bispectrum for the longitudinal mode for mme
def compute_P_B_mme_3h(k):
    return P_B_par(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pe, interp_pM, bispectrum_3h, P_L_interp)

# 3-halo contribution to the bispectrum for the longitudinal mode for mmg
def compute_P_B_mmg_3h(k):
    return P_B_par(np.array([0, 0, k]), cosmo, a, aHf, M, nM_vals, bM_vals, interp_pg, interp_pM, bispectrum_3h, P_L_interp)

#%%
# Test for a single k value

#%%

test_k_1 = 1e-2
test_k_2 = 1e1

P_test_T_1h_par = compute_P_T_1h_par(test_k_2)
print(f"P_T_1h_par = {P_test_T_1h_par:.3e}")

P_test_T_4h_par = compute_P_T_4h_par(test_k_1)
print(f"P_T_4h_par = {P_test_T_4h_par:.3e}")

P_test_B_mme_1h = compute_P_B_mme_1h(test_k_2)
print(f"P_B_mme_1h = {P_test_B_mme_1h:.3e}")

P_test_B_mme_3h = compute_P_B_mme_3h(test_k_1)
print(f"P_B_mme_3h = {P_test_B_mme_3h:.3e}")

P_test_B_mmg_1h = compute_P_B_mmg_1h(test_k_2)
print(f"P_B_mmg_1h = {P_test_B_mmg_1h:.3e}")

P_test_B_mmg_3h = compute_P_B_mmg_3h(test_k_1)
print(f"P_B_mmg_3h = {P_test_B_mmg_3h:.3e}")

P_test_T_4h_perp = compute_P_T_4h_perp(test_k_1)
print(f"P_T_4h_perp = {P_test_T_4h_perp:.3e}")

#%%
# Run over full k range

#%%

import concurrent.futures

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_T_4h_perp_vals = list(executor.map(compute_P_T_4h_perp, k_vals))
    
print("P_T_4h_perp_vals=", P_T_4h_perp_vals)

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_T_1h_vals_par = list(executor.map(compute_P_T_1h_par, k_vals))
    
print("P_T_1h_vals=", P_T_1h_vals_par)


with concurrent.futures.ProcessPoolExecutor() as executor:
    P_T_4h_vals_par = list(executor.map(compute_P_T_4h_par, k_vals))
    
print("P_T_4h_vals=", P_T_4h_vals_par)


with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mme_vals_3h = list(executor.map(compute_P_B_mme_3h, k_vals))
    
print("P_B_mme_3h_vals=", P_B_mme_vals_3h)


with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mme_vals_1h = list(executor.map(compute_P_B_mme_1h, k_vals))
    
print("P_B_mme_1h_vals=", P_B_mme_vals_1h)


with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mmg_vals_3h = list(executor.map(compute_P_B_mmg_3h, k_vals))
    
print("P_B_mmg_3h_vals=", P_B_mmg_vals_3h)


with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mmg_vals_1h = list(executor.map(compute_P_B_mmg_1h, k_vals))
    
print("P_B_mmg_1h_vals=", P_B_mmg_vals_1h)
