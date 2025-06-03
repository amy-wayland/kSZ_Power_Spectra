import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import kSZ_Integrals as ksz
import matplotlib.pyplot as plt
from scipy.integrate import simps

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
# Cross-correlations

#%%

# Matter-matter power spectrum
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=np.log(k_vals), a_arr=a_arr)

# Galaxy-matter power spectrum
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter power spectrum
pk_em = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy power spectrum
pk_eg = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg, lk_arr=lk_arr, a_arr=a_arr) # with two-halo term
pk_eg_1h = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg, lk_arr=lk_arr, a_arr=a_arr, get_2h=False) # one-halo term only

# Electron-electron power spectrum
pk_ee = (1/rho_mean**2) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=profile_density, lk_arr=lk_arr, a_arr=a_arr)

# Matter-matter power spectrum calculated via HaloProfiles.py
profile_baryon = hp.HaloProfileNFWBaryon(mass_def=hmd_200m, concentration=cM, **profile_parameters)
pk_mm_nfw_bar = ccl.halos.halomod_Pk2D(cosmo, hmc, profile_baryon, lk_arr=lk_arr, a_arr=a_arr)

#%%
# Calculate the 3D power spectra for the perpendicular and parallel components

#%%

# Initialise a kSZ object using kSZ_Integrals.py
kSZ_object = ksz.kSZIntegral(cosmo=cosmo, k_arr=k_vals, k_prime_arr=k_prime_vals, a_arr=a_arr, H=H)

# 3D power spectra for the perpendicular component
P_of_k_2d_perp = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1 = np.array([kSZ_object.integral_perp_1(k, pk_mm, pk_eg, i, a) for k in k_vals])
    P2 = np.array([kSZ_object.integral_perp_2(k, pk_em, pk_gm, i, a) for k in k_vals])
    P_of_k_2d_perp[:, i] = P1 - P2

pk2d_perp = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp, is_logp=False)

# 3D power specrtra for the parallel component
P_of_k_2d_par = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_par = np.array([kSZ_object.integral_par_1(k, pk_mm, pk_eg, i, a) for k in k_vals])
    P2_par = np.array([kSZ_object.integral_par_2(k, pk_em, pk_gm, i, a) for k in k_vals])
    P_of_k_2d_par[:, i] = P1_par - P2_par

pk2d_par = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par, is_logp=False)

#%%
# Create custom tracers to calculate the angular power spectrum

#%%

sigma_T_cgs = 6.65e-25 # cm^2
n_e0_cgs = 2e-7 # cm^{-3}
cm_per_Mpc = 3.0857e24 # cm / Mpc
sigma_T = sigma_T_cgs / cm_per_Mpc**2 # Mpc^2
n_e0 = n_e0_cgs * cm_per_Mpc**3 # Mpc^{-3}

z = (1 / a_arr) - 1
sorted_indices = np.argsort(z)
z = z[sorted_indices]
a_arr = a_arr[sorted_indices]

pz = (1 / np.sqrt(2 * np.pi * 0.05**2)) * np.exp(- 0.5 * ((z - 0.1) / 0.05)**2)
Hz = ccl.h_over_h0(cosmo, 1/(1+z)) * cosmo['h'] * 100
nz = Hz * pz

kernel_pi = ccl.get_density_kernel(cosmo, dndz=(z,nz))

chi = ccl.comoving_radial_distance(cosmo, 1/(1+z)) # Comoving distance in Mpc
dchi_dz = np.gradient(chi, z) # The CCL kernel takes a function of comoving distance weighted by the differential path length
weight_T = sigma_T * n_e0 / a_arr**2 * dchi_dz  # Dimensionless kernel

gc_pi = ccl.Tracer()
gc_T = ccl.Tracer()

gc_pi.add_tracer(cosmo, kernel=kernel_pi, der_bessel=-1, der_angles=0)
gc_T.add_tracer(cosmo, kernel=(a_arr, weight_T), der_bessel=-1, der_angles=0)

# Calculate the angular power spectra
ells = np.geomspace(2, 1000, 20)
C_ells = ccl.angular_cl(cosmo, gc_pi, gc_T, ells, p_of_k_a=pk2d_perp)
D_ells = ells * (ells + 1) * C_ells / (2 * np.pi)
C_ells_par = ccl.angular_cl(cosmo, gc_pi, gc_T, ells, p_of_k_a=pk2d_par)
D_ells_par = ells * (ells + 1) * C_ells_par / (2 * np.pi)

#%%
# Plot the angular power spectra

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

plt.plot(ells, D_ells, color="tab:blue", label=r'$D_{\ell, \perp}$')
plt.plot(ells, D_ells_par, color="tab:red", label=r'$D_{\ell, \parallel}$')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$D_{\ell} = \ell (\ell + 1) \, C_{\ell}^{\pi T} / (2\pi)$', fontsize=16)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=18, frameon=False, loc="lower left")
plt.savefig('kSZ_angular_power_spectrum.pdf',  format="pdf", bbox_inches="tight")
plt.show()