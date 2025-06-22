import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.interpolate import RectBivariateSpline, interp1d

#%%
# Set-up

#%%

# Define the cosmology
COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}

cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

# Scale factor and associated quantities
a = 1/(1+0.55)
H = cosmo['h']/ccl.h_over_h0(cosmo, a)
a_dot = a * H
D = ccl.growth_factor(cosmo, a)
ln_a = np.log(a)
ln_D = np.log(D)
f = ln_D / ln_a # np.gradient(ln_D, ln_a)

# Mass definition
hmd_200m = ccl.halos.MassDef200m

# Concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

# Mass function
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)

# Halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

# NFW profile to characterise the matter density around halos
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

# HOD model to characterise galaxy overdensity
pg = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)

# HMCalculator object for the mass integrals
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=10., nM=32)
hmc.precision['padding_lo_fftlog'] = 1e-2
hmc.precision['padding_hi_fftlog'] = 1e2
hmc.precision['n_per_decade'] = 1000

# Electron density profile
profile_parameters = {"lMc": 14.0, "beta": 1.0, "eta_b": 0.05}
pe = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)

pe.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)
pM.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-3.0)
pg.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)

# Normalisation to convert electron density to electron overdensity
rho_crit = ccl.physical_constants.RHO_CRITICAL
rho_bar = cosmo["Omega_b"] * rho_crit
log10M = np.linspace(11, 15, 128)
M = 10**log10M
n_M = nM(cosmo, M, a)
f_bound, f_ejected, f_star = pe._get_fractions(cosmo, M)
integrand = M * n_M * f_star
rho_star = np.trapz(integrand, log10M) / a**3
rho_mean = rho_bar + rho_star

#%%
# Precompute and interpolate Fourier transforms

#%%

# Define grids for M and k
log10M_grid = np.linspace(11, 15, 128)
M_grid = 10**log10M_grid
k_grid = np.logspace(-3, 2, 256)
log10k_grid = np.log10(k_grid)

# Arrays for Fourier transforms
u_pg_grid = np.zeros((len(k_grid), len(M_grid)))
u_pe_grid = np.zeros((len(k_grid), len(M_grid)))
u_pM_grid = np.zeros((len(k_grid), len(M_grid)))

# Fill the grids
for i, k_val in enumerate(k_grid):
    for j, M_val in enumerate(M_grid):
        u_pg_grid[i, j] = pg.fourier(cosmo, k_val, M_val, a) / M_val
        u_pe_grid[i, j] = pe.fourier(cosmo, k_val, M_val, a) / rho_mean
        u_pM_grid[i, j] = pM.fourier(cosmo, k_val, M_val, a) / M_val

interp_pg = RectBivariateSpline(log10k_grid, log10M_grid, u_pg_grid)
interp_pe = RectBivariateSpline(log10k_grid, log10M_grid, u_pe_grid)
interp_pM = RectBivariateSpline(log10k_grid, log10M_grid, u_pM_grid)


#%%
# Functions to calculate the tree-level bispectrum

#%%

k_dense = np.logspace(-3, 1, 1000)
P_dense = ccl.linear_matter_power(cosmo, k_dense, a)
P_L_interp = interp1d(k_dense, P_dense, bounds_error=False, fill_value=0.0)


def P_L(k_vec):
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


def B_tree_level(k_1_vec, k_2_vec, k_3_vec):
    
    B_13 = 2 * F_2(k_1_vec, k_2_vec) * P_L(k_1_vec) * P_L(k_2_vec)
    B_23 = 2 * F_2(k_2_vec, k_3_vec) * P_L(k_2_vec) * P_L(k_3_vec)
    B_31 = 2 * F_2(k_3_vec, k_1_vec) * P_L(k_3_vec) * P_L(k_1_vec)
    
    return B_13 + B_23 + B_31


#%%
# Functions to pre-compute halo propertiess

#%%

def precompute_nM(cosmo, M_vals, a):
    """
    Precompute n(M) at given mass values
    """
    nM_vals = nM(cosmo, M_vals, a)
    return nM_vals


def precompute_bM(cosmo, M_vals, a):
    """
    Precompute b(M) at given mass values
    """
    bM_vals = bM(cosmo, M_vals, a)
    return bM_vals


#%%
# Functions to calculate the three-halo and one-halo contributions

#%%

def bispectrum_3h(k, kp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pa, interp_pM):
    
    k1 = k
    k2 = -kp
    k3 = - (k - kp)
    ks = [k1, k2, k3]
    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)
    dlog10M = np.log(10) * M_vals  # dM = ln(10) * M * dlog10M

    # Evaluate each profile over (k, M) grid
    u1_vals = interp_pM.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pM.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pa.ev(np.log10(k_norms[2]), log10M)

    # Compute separate integrals for each u_i
    integrand1 = dlog10M * nM_vals * bM_vals * u1_vals
    integrand2 = dlog10M * nM_vals * bM_vals * u2_vals
    integrand3 = dlog10M * nM_vals * bM_vals * u3_vals

    I1 = np.trapz(integrand1, log10M)
    I2 = np.trapz(integrand2, log10M)
    I3 = np.trapz(integrand3, log10M)

    B_tree = B_tree_level(k1, k2, k3)

    result = B_tree * I1 * I2 * I3 / rho_mean**3
    #print(f"bispectrum_3h = {result:.3e}")
    return result


def bispectrum_1h(k, kp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pa, interp_pM):
    
    k1 = k
    k2 = -kp
    k3 = - (k - kp)
    ks = [k1, k2, k3]
    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)
    dlog10M = np.log(10) * M_vals  # dM = ln(10) * M * dlog10M

    # Evaluate each profile over (k, M) grid
    u1_vals = interp_pM.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pM.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pa.ev(np.log10(k_norms[2]), log10M)

    integrand = dlog10M * nM_vals * bM_vals * u1_vals * u2_vals * u3_vals
    integral = np.trapz(integrand, log10M)
    result = integral / rho_mean
    #print(f"bispectrum_1h = {result:.3e}")
    return result

#%%
# Define a function to calculate the final integral

#%%

def P_B(k_vec, cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pa, interp_pM, bispectrum_func, n_k=10, n_mu=10, n_phi=16):
    
    k_mag = np.linalg.norm(k_vec)
    if k_mag < 1e-5:
        return 0.0

    # Integration grids
    k_primes = np.logspace(-3, 1, n_k)
    mu_primes = np.linspace(-0.99, 0.99, n_mu)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)

    dk = np.gradient(k_primes)
    dmu = 2 / n_mu
    dphi = 2 * np.pi / n_phi

    prefactor = (a_dot * f)**2 / ((2 * np.pi)**3 * k_mag**2)
    result = 0.0

    for i, kp in enumerate(k_primes):
        for mu in mu_primes:
            sin_theta = np.sqrt(1 - mu**2)

            # k' vectors over phi
            kp_vecs = kp * np.stack([sin_theta * cos_phis, sin_theta * sin_phis, mu * np.ones_like(phis)], axis=1)

            # Evaluate bispectrum for each phi
            B_vals = np.array([
                bispectrum_func(k_vec, kp_vec, cosmo, a, M_vals, nM_vals, bM_vals, interp_pa, interp_pM)
                for kp_vec in kp_vecs])

            block_sum = np.sum(k_mag * kp * mu * B_vals * dk[i] * dmu * dphi)
            result += block_sum

    final_result = prefactor * result
    return final_result


#%%
# Calculations

#%%

M_vals = np.logspace(11, 15, 128)
nM_vals = precompute_nM(cosmo, M_vals, a)
bM_vals = precompute_bM(cosmo, M_vals, a)
k_vals = np.logspace(-3, 1, 128)

#%%

import concurrent.futures

# For B_{mme}

def compute_P_B_mme_3h(k):
    return P_B(np.array([0, 0, k]), cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pe, interp_pM, bispectrum_3h)

def compute_P_B_mme_1h(k):
    return P_B(np.array([0, 0, k]), cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pe, interp_pM, bispectrum_1h)

# For B_{mmg}

def compute_P_B_mmg_3h(k):
    return P_B(np.array([0, 0, k]), cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pg, interp_pM, bispectrum_3h)

def compute_P_B_mmg_1h(k):
    return P_B(np.array([0, 0, k]), cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pg, interp_pM, bispectrum_1h)

#%%

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mme_vals_3h = list(executor.map(compute_P_B_mme_3h, k_vals))
    
print("P_3h_vals=", P_B_mme_vals_3h)

#%%

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mme_vals_1h = list(executor.map(compute_P_B_mme_1h, k_vals))
    
print("P_1h_vals=", P_B_mme_vals_1h)

#%%

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mmg_vals_3h = list(executor.map(compute_P_B_mmg_3h, k_vals))
    
print("P_3h_vals=", P_B_mmg_vals_3h)

#%%

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_B_mmg_vals_1h = list(executor.map(compute_P_B_mmg_1h, k_vals))
    
print("P_1h_vals=", P_B_mmg_vals_1h)

#%%

# Test
test_k = 1e-2
P_test = compute_P_B_mme_3h(test_k)
print(f"P_B_mme(k={test_k:.2e}) = {P_test:.3e}")
