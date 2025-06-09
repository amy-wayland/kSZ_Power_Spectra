import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt

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
             #"transfer_function": "eisenstein_hu"}
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

# Scale factor and associated quantities
a = 1/(1+0.55)
H = ccl.h_over_h0(cosmo, a) * cosmo['h'] * 100  # km/s/Mpc
a_dot = a * H
D = ccl.growth_factor(cosmo, a)
ln_a = np.log(a)
ln_D = np.log(D)
f = ln_D / ln_a #np.gradient(ln_D, ln_a)

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
hmc.precision['n_per_decade'] = 2000

# Electron density profile
profile_parameters = {"lMc": 14.0, "beta": 1.0, "eta_b": 0.5}
pe = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
pe.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)

# Normalisation to convert electron density to electron overdensity
rho_crit = ccl.physical_constants.RHO_CRITICAL
rho_bar = cosmo["Omega_b"] * rho_crit
log10M = np.linspace(10, 15, 1000)
M = 10**log10M
n_M = nM(cosmo, M, a)
f_bound, f_ejected, f_star = pe._get_fractions(cosmo, M)
integrand = M * n_M * f_star
rho_star = np.trapz(integrand, log10M) / a**3
rho_mean = rho_bar + rho_star


#%%
# Functions to calculate the tree-level trispectrum

#%%

def F_2(k_1_vec, k_2_vec):
    
    k_1_norm = np.linalg.norm(k_1_vec)
    k_2_norm = np.linalg.norm(k_2_vec)
    dot_prod = np.dot(k_1_vec, k_2_vec)
    
    return 5/7 + (1/2) * (dot_prod / (k_1_norm * k_2_norm)) * ((k_1_norm/k_2_norm) + (k_2_norm/k_1_norm)) + (2/7)*(dot_prod / (k_1_norm * k_2_norm))**2


def G_2(k_1_vec, k_2_vec):
    
    k_1_norm = np.linalg.norm(k_1_vec)
    k_2_norm = np.linalg.norm(k_2_vec)
    dot_prod = np.dot(k_1_vec, k_2_vec)
    
    return 3/7 + (1/2) * (1/k_1_norm**2 + 1/k_2_norm**2) * dot_prod + (4/7) * ((dot_prod) / (k_1_norm * k_2_norm))**2


def Q(k_1_vec, k_2_vec, k_3_vec):
    
    k_123_vec = k_1_vec + k_2_vec + k_3_vec
    k_23_vec = k_2_vec + k_3_vec
    k_1_norm = np.linalg.norm(k_1_vec)
    k_123_norm = np.linalg.norm(k_123_vec)
    k_23_norm = np.linalg.norm(k_23_vec)
    
    prefactor_1 = (7 *np.dot(k_123_vec, k_1_vec)) / k_1_norm**2
    term_1 = prefactor_1 * F_2(k_2_vec, k_3_vec)
    prefactor_2 = (7 * k_123_vec * k_23_vec) / k_23_norm**2 + (2 * k_123_norm**2 * np.dot(k_23_vec, k_1_vec)) / (k_23_norm**2 * k_1_norm**2)
    term_2 = prefactor_2 * G_2(k_2_vec, k_3_vec)
    
    return term_1 + term_2


def F_3(k_1_vec, k_2_vec, k_3_vec):
    return (1/54) * (Q(k_1_vec, k_2_vec, k_3_vec) + Q(k_2_vec, k_3_vec, k_1_vec) + Q(k_3_vec, k_1_vec, k_2_vec))


def P_L(k_vec, cosmo=cosmo, a=1.0):
    k_norm = np.linalg.norm(k_vec)
    return ccl.linear_matter_power(cosmo, k_norm, a)


def T_1122(k_vec, k_prime_vec):
    
    k_diff_vec = k_vec - k_prime_vec
    k_diff_2_vec = k_vec - 2 * k_prime_vec
    k_norm = np.linalg.norm(k_vec)
    k_prime_norm = np.linalg.norm(k_prime_vec)
    k_diff_norm = np.linalg.norm(k_diff_vec)
    k_diff_2_norm = np.linalg.norm(k_diff_2_vec)
    
    term_1 = F_2(k_vec, -k_diff_vec) * F_2(k_vec, -k_prime_vec) * P_L(k_norm) * P_L(k_diff_norm) * P_L(k_prime_norm)
    term_2 = F_2(k_diff_2_vec, -k_diff_vec) * P_L(k_diff_2_norm) * P_L(k_diff_norm) * P_L(k_prime_norm)
    
    return 16 * (term_1 + term_2)


def T_1113(k_vec, k_prime_vec):
    
    k_diff_vec = k_vec - k_prime_vec
    k_prime_norm = np.linalg.norm(k_prime_vec)
    k_diff_norm = np.linalg.norm(k_diff_vec)
    
    prefactor = 12 * P_L(k_diff_norm) * P_L(k_prime_norm)**2
    
    return prefactor * (F_3(k_diff_vec, -k_prime_vec, k_prime_vec) + F_3(k_prime_vec, -k_diff_vec, k_diff_vec))


def T_tree_level(k_vec, k_prime_vec):
    return T_1122(k_vec, k_prime_vec) + T_1113(k_vec, k_prime_vec)


#%%
# Functions to calculate the one-halo and four-halo contributions

#%%

def trispectrum_1h(k_vec, k_prime_vec):
    
    k_diff = k_vec - k_prime_vec
    k_norms = [np.linalg.norm(k_diff), np.linalg.norm(k_prime_vec)]
    
    def log_integrand(logM):
        M = M = 10**logM
        u1 = pg.fourier(cosmo, k_norms[0], M, a)
        u2 = pe.fourier(cosmo, k_norms[1], M, a) / rho_mean
        u3 = pM.fourier(cosmo, k_norms[1], M, a)
        u4 = pM.fourier(cosmo, k_norms[0], M, a)
        return M * nM(cosmo, M, a) * bM(cosmo, M, a) * u1 * u2 * u3 * u4

    logM_vals = np.linspace(10, 15, 256)
    integrand_vals = [log_integrand(logM) for logM in logM_vals]

    return np.trapz(integrand_vals, logM_vals)
    

def trispectrum_4h(k_vec, k_prime_vec):
    
    k_diff = k_vec - k_prime_vec
    k_norms = [np.linalg.norm(k_diff), np.linalg.norm(k_prime_vec)]
    
    def log_integrand(logM):
        M = 10**logM
        u1 = pg.fourier(cosmo, k_norms[0], M, a)
        u2 = pe.fourier(cosmo, k_norms[1], M, a) / rho_mean
        u3 = pM.fourier(cosmo, k_norms[1], M, a)
        u4 = pM.fourier(cosmo, k_norms[0], M, a)
        return M * nM(cosmo, M, a) * bM(cosmo, M, a) * u1 * u2 * u3 * u4
    
    prefactor = T_tree_level(k_vec, k_prime_vec)
    logM_vals = np.linspace(10, 15, 256)
    integrand_vals = [log_integrand(logM) for logM in logM_vals]
    
    return prefactor * np.trapz(integrand_vals, logM_vals)


#%%
# Define a function to calculate the connected non-Gaussian power spectrum

#%%

def P_cNG(k, cosmo, a, f, a_dot, trispectrum_func, k_prime_min=1e-3, k_prime_max=1e1):
    
    k_prime_vals = np.linspace(k_prime_min, k_prime_max, 32)
    mu_vals = np.linspace(-1, 1, 32)
    phi_vals = np.linspace(0, 2 * np.pi, 32)
    
    prefactor = (a_dot * f)**2 / (2 * np.pi)**6
    
    k_vec = np.array([0, 0, k]) # k vector defined to be along the z-axis
    K_prime, MU, PHI = np.meshgrid(k_prime_vals, mu_vals, phi_vals, indexing='ij') # 3D meshgrid for k', mu, phi

    # Compute k' vector components assuming k along z axis
    k_prime_x = K_prime * np.sqrt(1 - MU**2) * np.cos(PHI)
    k_prime_y = K_prime * np.sqrt(1 - MU**2) * np.sin(PHI)
    k_prime_z = K_prime * MU

    # Compute k_prime_vec at each point
    k_prime_vec = np.stack([k_prime_x, k_prime_y, k_prime_z], axis=-1)

    # Compute denominator: k^2 + k'^2 - 2 k k' mu
    denom = k**2 + K_prime**2 - 2 * k * K_prime * MU
    denom = np.where(denom == 0, 1e-15, denom)

    factor = (1 - MU**2) / denom

    # Now evaluate trispectrum T at each point
    k_prime_vec_flat = k_prime_vec.reshape(-1, 3) # Flatten arrays for looping
    T_vals = np.empty(k_prime_vec_flat.shape[0])

    for i in range(k_prime_vec_flat.shape[0]):
        T_vals[i] = trispectrum_func(k_vec, k_prime_vec_flat[i])

    # Reshape T_vals back to (N_k_prime, N_mu, N_phi)
    T_vals = T_vals.reshape(32, 32, 32)

    # Full integrand:
    integrand = (K_prime**2) * factor * T_vals

    # Integrate over phi (last axis)
    int_phi = np.trapz(integrand, phi_vals, axis=2)

    # Integrate over mu (middle axis)
    int_mu = np.trapz(int_phi, mu_vals, axis=1)

    # Integrate over kprime (first axis)
    integral = np.trapz(int_mu, k_prime_vals, axis=0)

    return prefactor * integral

    
#%%
# Calculate the 1-halo and 4-halo contributions

#%%

k_small = np.logspace(-1, 1, 30)   # small scales: 0.1 to 10
k_large = np.logspace(-3, -1, 30)  # large scales: 0.001 to 0.1

P_1h_vals = np.array([P_cNG(k, cosmo, a, f, a_dot, trispectrum_1h, k_prime_min=0.1, k_prime_max=20) for k in k_small])
P_4h_vals = np.array([P_cNG(k, cosmo, a, f, a_dot, trispectrum_4h, k_prime_min=1e-3, k_prime_max=1)for k in k_large])

#%%

k_small = 0.1
k_large = 0.001

P_1h_vals = P_cNG(k_small, cosmo, a, f, a_dot, trispectrum_1h, k_prime_min=0.1, k_prime_max=20)
P_4h_vals = P_cNG(k_large, cosmo, a, f, a_dot, trispectrum_4h, k_prime_min=1e-3, k_prime_max=1)

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

plt.loglog(k_small, P_1h_vals, label='1-halo (small scales)', color='tab:blue')
plt.loglog(k_large, P_4h_vals, label='4-halo (large scales)', color='tab:red')

plt.xlabel(r'$k\, [h/\mathrm{Mpc}]$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\rm cNG}(k)$', fontsize=20)
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.savefig('kSZ_cNG_power_spectrum.pdf', format="pdf", bbox_inches="tight")
plt.show()
