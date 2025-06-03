import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
import kSZ_Integrals as ksz
from scipy.integrate import simps
from scipy.special import erf 

#%%
# Set-up

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

k_vals = np.logspace(-4, 2, 256)
lk_arr = np.log(k_vals)
k_prime_vals = np.logspace(-4, 2, 256)
a_arr = np.linspace(0.1, 1, 32)
H = ccl.h_over_h0(cosmo, a_arr) * cosmo['h'] * 100 # km/s/Mpc

#%%
# Halo model implementation

#%%

# Mass definition with Delta = 200 times the matter density
hmd_200m = ccl.halos.MassDef200m

# Duffy 2008 concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

# Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)

# Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

# NFW profile to characterise the matter density around halos
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

# HOD model to characterise galaxy overdensity
pg = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)

# HMCalculator object for the mass integrals
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=10., nM=32)

#%%
# Normalise the electron profile

#%%

profile_parameters = {"lMc": 14.0, "beta": 1.0, "eta_b": 0.5}
profile_density = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
profile_density.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)

# Contribution from baryons
rho_crit = ccl.physical_constants.RHO_CRITICAL
rho_bar = cosmo["Omega_b"] * rho_crit

# Contribution from stars
a = 1.0
log10M = np.linspace(10, 15, 1000)
M = 10**log10M
n_M = nM(cosmo, M, a)
f_bound, f_ejected, f_star = profile_density._get_fractions(cosmo, M)
integrand = M * n_M * f_star
rho_star = simps(integrand, log10M) / a**3

# Normalisation factor to convert the electron density into the electron overdensity
rho_mean = rho_bar + rho_star

#%%
# Functions to calculate the central and satellite galaxy profiles

#%%

def N_c(M, M_min):
    '''
    Returns the mean number of central galaxies

    '''
    sig_lnM = 0.4
    return 0.5 * (1 + erf((np.log10(M / M_min)) / sig_lnM))


def N_s(M, M_0, M_1, alpha=1.0):
    '''
    Returns the mean number of satellite galaxies
    
    '''
    return np.heaviside(M - M_0, 0) * ((M - M_0) / M_1)**(alpha)


def mean_halo_mass(M, nM, N_c, N_s):
    '''
    Returns the mean halo mass
    
    '''
    log10_M = np.log10(M)
    N_g = N_c + N_s
    integrand_1 = M * nM * N_g
    integrand_2 = nM * N_g
    return simps(integrand_1, log10_M) / simps(integrand_2, log10_M)
    
#%%
# Central galaxy only calculation

#%%

a = 1.0
log10_M = np.linspace(10, 15, 1000)
M = 10**log10_M
n_M = nM(cosmo, M, a)

M_0 = 1e15
log10_M_1 = 13.0
M_1 = 10**(log10_M_1)
M_min_cen = 2.03e12 # value of M_min at which M_mean = 1e13

N_cen = N_c(M, M_min_cen)
N_sat = N_s(M, M_0, M_1)

M_mean = mean_halo_mass(M, n_M, N_cen, N_sat)
print(f"{M_mean:.2e}")

# The HOD model to characterise galaxy overdensity for central galaxies only
pg_cen = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(M_min_cen), log10M0_0=15.0, log10M1_0=13.0, alpha_0=1.2)

#%%
# Now with satellite galaxies

#%%

M_0 = 1e11
M_1 = 3e12
M_min_sat = 3.89e10

N_cen = N_c(M, M_min_sat)
N_sat = N_s(M, M_0, M_1)

M_mean = mean_halo_mass(M, n_M, N_cen, N_sat)
print(f"{M_mean:.2e}")

# The HOD model to characterise galaxy overdensity now with satellite galaxies
pg_sat = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(M_min_sat), log10M0_0=11.0, log10M1_0=np.log10(3e12), alpha_0=1.2)

#%%
# Cross-correlations

#%%

# Matter-matter power spectrum
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=np.log(k_vals), a_arr=a_arr)

# Galaxy-matter power spectrum
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pg_cen, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter power spectrum
pk_em = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy power spectrum
pk_eg = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg_cen, lk_arr=lk_arr, a_arr=a_arr) # with two-halo term
pk_eg_1h = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg_cen, lk_arr=lk_arr, a_arr=a_arr, get_2h=False) # one-halo term only

# Electron-electron power spectrum
pk_ee = (1/rho_mean**2) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=profile_density, lk_arr=lk_arr, a_arr=a_arr)

# Matter-matter power spectrum calculated via HaloProfiles.py
profile_baryon = hp.HaloProfileNFWBaryon(mass_def=hmd_200m, concentration=cM, **profile_parameters)
pk_mm_nfw_bar = ccl.halos.halomod_Pk2D(cosmo, hmc, profile_baryon, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-matter power spectrum with satellite galaxies
pk_gm_sat = ccl.halos.halomod_Pk2D(cosmo, hmc, pg_sat, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy power spectrum with satellite galaxies
pk_eg_sat = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg_sat, lk_arr=lk_arr, a_arr=a_arr) # with two-halo term
pk_eg_1h_sat = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg_sat, lk_arr=lk_arr, a_arr=a_arr, get_2h=False) # one-halo term only

#%%

plt.plot(k_vals, pk_eg(k_vals, 1.0), label=r'central galaxy only', color="tab:red")
plt.plot(k_vals, pk_eg_sat(k_vals, 1.0), label=r'central + satellite galaxies', color="tab:blue")
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{\rm eg}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False, loc='lower left')
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_investigating_satellite_galaxies.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Now perform the calculations

#%%

# Initialise a kSZ object using kSZ_Integrals.py
kSZ_object = ksz.kSZIntegral(cosmo=cosmo, k_arr=k_vals, k_prime_arr=k_prime_vals, a_arr=a_arr, H=H)

# Calculate the integral for different values of k and a for the central galaxy only
P_of_k_2d_cen = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1 = np.array([kSZ_object.integral_perp_1(k, pk_mm, pk_eg, i, a) for k in k_vals])
    P2 = np.array([kSZ_object.integral_perp_2(k, pk_em, pk_gm, i, a) for k in k_vals])
    P_of_k_2d_cen[:, i] = P1 - P2
    
# Calculate the integral for different values of k and a now with satellite galaxies
P_of_k_2d_sat = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1 = np.array([kSZ_object.integral_perp_1(k, pk_mm, pk_eg_sat, i, a) for k in k_vals])
    P2 = np.array([kSZ_object.integral_perp_2(k, pk_em, pk_gm_sat, i, a) for k in k_vals])
    P_of_k_2d_sat[:, i] = P1 - P2

#%%
# Plot the results

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

plt.plot(k_vals, P_of_k_2d_cen, label=r'central galaxy only', color='tab:red')
plt.plot(k_vals, P_of_k_2d_sat, label=r'with satellite galaxies', color='tab:blue')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=18)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=18)
plt.loglog()
plt.legend(fontsize=11, frameon=False, loc="upper right")
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.show()