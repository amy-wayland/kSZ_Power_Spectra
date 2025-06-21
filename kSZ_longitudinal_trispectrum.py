import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
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
# Functions to calculate the tree-level trispectrum

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
    

def T_1122(k, kp, kpp):
    
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
        P1 = P_L(q_vec)
        P2 = P_L(ks[i])
        P3 = P_L(ks[j])
        total += F2_1 * F2_2 * P1 * P2 * P3

    return 4 * total


def T_1113(k, kp, kpp):
    
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]

    total = 0.0
    indices = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]

    for i, j, l in indices:
        F3_val = F_3(ks[i], ks[j], ks[l])
        P1 = P_L(ks[i])
        P2 = P_L(ks[j])
        P3 = P_L(ks[l])
        total += F3_val * P1 * P2 * P3

    return 6 * total


def T_tree_level(k, kp, kpp):
    return T_1122(k, kp, kpp) + T_1113(k, kp, kpp)


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
# Function to calculate the four-halo and one-halo contributions

#%%

def trispectrum_4h(k, kp, kpp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM):
    
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]
    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)
    dlog10M = np.log(10) * M_vals  # dM = ln(10) * M * dlog10M

    # Evaluate each profile over (k, M) grid
    u1_vals = interp_pg.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pe.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pM.ev(np.log10(k_norms[2]), log10M)
    u4_vals = interp_pM.ev(np.log10(k_norms[3]), log10M)

    # Compute separate integrals for each u_i
    integrand1 = dlog10M * nM_vals * bM_vals * u1_vals
    integrand2 = dlog10M * nM_vals * bM_vals * u2_vals
    integrand3 = dlog10M * nM_vals * bM_vals * u3_vals
    integrand4 = dlog10M * nM_vals * bM_vals * u4_vals

    I1 = np.trapz(integrand1, log10M)
    I2 = np.trapz(integrand2, log10M)
    I3 = np.trapz(integrand3, log10M)
    I4 = np.trapz(integrand4, log10M)

    T_tree = T_tree_level(k, kp, kpp)

    result = T_tree * I1 * I2 * I3 * I4 / rho_mean**4
    #print(f"trispectrum_4h = {result:.3e}")
    return result


def trispectrum_1h(k, kp, kpp, cosmo, a, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM):
    
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

    dlog10M = np.log(10) * M_vals
    integrand = dlog10M * nM_vals * bM_vals * u1 * u2 * u3 * u4
    integral = np.trapz(integrand, log10M)
    result = integral / rho_mean
    #print(f"trispectrum_1h = {result:.3e}")
    return result
    
    
#%%
# Define a function to calculate the connected non-Gaussian power spectrum

#%%

def P_cng_par(k_vec, cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_func, n_k=10, n_mu=10, n_phi=16):

    k_mag = (np.dot(k_vec, k_vec))**(1/2)
    if k_mag < 1e-5:
        return 0.0

    # Integration grids
    k_primes = np.logspace(-3, 1, n_k)
    kpp_primes = np.logspace(-3, 1, n_k)
    mu_primes = np.linspace(-0.99, 0.99, n_mu)
    mupp_primes = np.linspace(-0.99, 0.99, n_mu)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)

    dk = np.gradient(k_primes)
    dkp = np.gradient(kpp_primes)
    dmu = 2 / n_mu
    dphi = 2 * np.pi / n_phi

    prefactor = (a_dot * f)**2 / (2 * np.pi)**5
    result = 0.0

    for i, kp in enumerate(k_primes):
        for j, kpp in enumerate(kpp_primes):
            for mu1 in mu_primes:
                sin_theta1 = np.sqrt(1 - mu1**2)

                # Fixed k' vector
                kp_vec = kp * np.array([sin_theta1, 0.0, mu1])

                for mu2 in mupp_primes:
                    sin_theta2 = np.sqrt(1 - mu2**2)

                    # Vectorised k'' vectors
                    kpp_vecs = kpp * np.stack([sin_theta2 * cos_phis, sin_theta2 * sin_phis, mu2 * np.ones_like(phis)], axis=1)

                    # Evaluate trispectrum for each phi
                    T_vals = np.array([trispectrum_func(k_vec, kp_vec, kpp_vec, cosmo, a, M_vals, nM_vals, bM_vals,interp_pg, interp_pe, interp_pM) for kpp_vec in kpp_vecs])

                    block_sum = np.sum(kp * kpp * mu1 * mu2 * sin_theta1 * sin_theta2 * T_vals * dk[i] * dkp[j] * dmu * dmu * dphi)
                    result += block_sum

    final_result = prefactor * result / k_mag**2
    return final_result

    
#%%
# Calculations

#%%

M_vals = np.logspace(11, 15, 128)
nM_vals = precompute_nM(cosmo, M_vals, a)
bM_vals = precompute_bM(cosmo, M_vals, a)
k_vals_4h = np.logspace(-3, -2, 30)
k_vals_1h = np.logspace(0, 1, 30)

#%%

import concurrent.futures

def compute_P_4h_par(k):
    return P_cng_par(np.array([0, 0, k]), cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_4h)

def compute_P_1h_par(k):
    return P_cng_par(np.array([0, 0, k]), cosmo, a, a_dot, f, M_vals, nM_vals, bM_vals, interp_pg, interp_pe, interp_pM, trispectrum_1h)

#%%

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_4h_vals_par = list(executor.map(compute_P_4h_par, k_vals_4h))
    
print("P_4h_vals=", P_4h_vals_par)


#%%

with concurrent.futures.ProcessPoolExecutor() as executor:
    P_1h_vals_par = list(executor.map(compute_P_1h_par, k_vals_1h))
    
print("P_1h_vals=", P_1h_vals_par)

#%%

# Test for the four-halo calculation
test_k_4h = 1e-2
P_test_4h = compute_P_4h_par(test_k_4h)
print(f"P_cNG_4h(k={test_k_4h:.2e}) = {P_test_4h:.3e}")

#%%

# Test for the one-halo calculation
test_k_1h = 1e2
P_test_1h = compute_P_1h_par(test_k_1h)
print(f"P_cNG_1h(k={test_k_1h:.2e}) = {P_test_1h:.3e}")

#%%

# Check that the four-halo term dominates on large scales
P_test_1h_ls = compute_P_1h_par(test_k_4h)
print(f"P_cNG_1h_ls(k={test_k_4h:.2e}) = {P_test_1h_ls:.3e}") 
# We find that the one-halo term is much smaller than the four-halo term on large scales

#%%

# Check that one-halo term dominates on small scales
P_test_4h_ss = compute_P_4h_par(test_k_1h)
print(f"P_cNG_4h_ss(k={test_k_1h:.2e}) = {P_test_4h_ss:.3e}")
# We find that the four-halo term vanishes on small scales