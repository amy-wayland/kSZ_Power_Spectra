import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt

#%%
# Functions to compute the integrals

#%%

def P_perp_1(k, k_prime_vals, P_of_k_1, P_of_k_2, a, aHf_arr, a_index):
    '''
    Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term for the perpendicular mode

    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_m \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_e}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    aHf = aHf_arr[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 256)

    def integrand(mu, k_prime):
        q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
        return aHf**2 * (1/(2 * np.pi)**2) * (1 - mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


def P_perp_2(k, k_prime_vals, P_of_k_1, P_of_k_2, a, aHf_arr, a_index):
    '''
    Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term for the perpendicular mode
    
    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_e \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_m}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    aHf = aHf_arr[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 256)
    
    def integrand(mu, k_prime):
        p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        q = np.sqrt(p)
        return aHf**2 * (1/(2 * np.pi)**2) * k_prime**2 * (1 - mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
    
    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)
   

def P_par_1(k, k_prime_vals, P_of_k_1, P_of_k_2, a, aHf_arr, a_index):
    '''
    Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term for the parallel mode

    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_m \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_e}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    aHf = aHf_arr[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 256)

    def integrand(mu, k_prime):
        q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
        return aHf**2 * (1/(2 * np.pi)**2) * mu**2 * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


def P_par_2(k, k_prime_vals, P_of_k_1, P_of_k_2, a, aHf_arr, a_index):
    '''
    Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term for the parallel mode
    
    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_e \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_m}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    aHf = aHf_arr[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 256)
    
    def integrand(mu, k_prime):
        p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        q = np.sqrt(p)
        return aHf**2 * (1/(2 * np.pi)**2) * (mu * k_prime * k) * (1 - mu * (k_prime/k)) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
    
    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.abs(np.trapz(vals, mu_vals))

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


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

log10M = np.linspace(11, 15, 1000)
M = 10**log10M

H_arr = cosmo['h'] * ccl.h_over_h0(cosmo, a_arr) / ccl.physical_constants.CLIGHT_HMPC
f_arr = cosmo.growth_rate(a_arr)

aHf_arr = a_arr * H_arr * f_arr

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
pg = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)

# Halo model integral calculator
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=11., nM=32)

# Electron density
profile_parameters = {"lMc": 10.0, "beta": 0.6, "eta_b": 0.05, "A_star": 0.0}
profile_density = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
profile_density.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)

# Normalisation to convert to electron overdensity
z = 0.55
a = 1/(1+z)
rho_crit = ccl.rho_x(cosmo, a, 'critical')
rho_bar = cosmo["Omega_b"] * rho_crit
n_M = nM(cosmo, M, a)
f_bound, f_ejected, f_star = profile_density._get_fractions(cosmo, M)
integrand = M * n_M * f_star
rho_star = np.trapz(integrand, log10M) / a**3
rho_mean = rho_bar - rho_star

# Check normalisation of the electron density profile
rho_gas_from_prof = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1e-4, a, profile_density)
print(rho_gas_from_prof/rho_mean)

#%%
# Cross-correlations

#%%

# Matter-matter
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=np.log(k_vals), a_arr=a_arr)

# Galaxy-matter
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter
pk_em = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy
pk_eg = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy one-halo term only
pk_eg_1h = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg, lk_arr=lk_arr, a_arr=a_arr, get_2h=False)

# Electron-electron
pk_ee = (1/rho_mean**2) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=profile_density, lk_arr=lk_arr, a_arr=a_arr)

#%%
# 3D power spectra calculations 

#%%

P_of_k_2d_perp = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_par = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    
    P1_perp = np.array([P_perp_1(k, k_prime_vals, pk_mm, pk_eg, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_perp[:, i] = P1_perp
    
    P1_par = np.array([P_perp_2(k, k_prime_vals, pk_mm, pk_eg, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_par[:, i] = P1_par
    
#%% 

z = (1 / a_arr) - 1
sorted_indices = np.argsort(a_arr)
z = z[sorted_indices]
a_arr = a_arr[sorted_indices]
H_arr = H_arr[sorted_indices]
f_arr = f_arr[sorted_indices]

P_of_k_2d = P_of_k_2d_perp[:, sorted_indices]
P_of_k_2d_par = P_of_k_2d_par[:, sorted_indices]

pk2d_perp = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp, is_logp=False)
pk2d_par = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par, is_logp=False)

#%%
# Create custom tracers to calculate the angular power spectrum

#%%

sigma_T_cgs = 6.65e-25 # cm^2
n_e0_cgs = 2e-7 # cm^{-3}
cm_per_Mpc = 3.0857e24 # cm / Mpc
sigma_T = (sigma_T_cgs / cm_per_Mpc**2) * cosmo['h']**2 # (Mpc/h)**2
n_e0 = (n_e0_cgs * cm_per_Mpc**3) / cosmo['h']**3 # 1/(Mpc/h)**3

pz = (1 / np.sqrt(2 * np.pi * 0.3**2)) * np.exp(- 0.5 * ((z - 0.55) / 0.3)**2)
Hz = ccl.h_over_h0(cosmo, 1/(1+z)) * cosmo['h'] * 100
nz = Hz * pz

sort_idx = np.argsort(z)
z = z[sort_idx]
pz = pz[sort_idx]

kernel_pi = ccl.get_density_kernel(cosmo, dndz=(z,pz))

chi = ccl.comoving_radial_distance(cosmo, 1/(1+z)) # comoving distance chi(z) in Mpc
dchi_dz = ccl.physical_constants.CLIGHT_HMPC*cosmo['h']/ccl.h_over_h0(cosmo, 1/(1+z))
weight_T = dchi_dz / a_arr**2  # dimensionless kernel

gc_pi = ccl.Tracer()
gc_T = ccl.Tracer()

gc_pi_par = ccl.Tracer()
gc_T_par = ccl.Tracer()

gc_pi.add_tracer(cosmo, kernel=kernel_pi, der_bessel=0, der_angles=0)
gc_T.add_tracer(cosmo, kernel=(chi, weight_T), der_bessel=0, der_angles=0)

gc_pi_par.add_tracer(cosmo, kernel=kernel_pi, der_bessel=1, der_angles=0)
gc_T_par.add_tracer(cosmo, kernel=(chi, weight_T), der_bessel=1, der_angles=0)

#%%
# Calculate the angular power spectra

#%%

ells = np.geomspace(1e1, 1e4, 100)
C_ells_perp = sigma_T * n_e0 * ((ells * (ells+1)) / (ells+1/2)**2) * ccl.angular_cl(cosmo, gc_pi, gc_T, ells, p_of_k_a=pk2d_perp) / 2
C_ells_par = sigma_T * n_e0 * ((ells * (ells+1)) / (ells+1/2)**2) * ccl.angular_cl(cosmo, gc_pi_par, gc_T_par, ells, p_of_k_a=pk2d_par) / 4

D_ells_perp = ells * (ells + 1) * C_ells_perp / (2 * np.pi)
D_ells_par = ells * (ells + 1) * C_ells_par / (2 * np.pi)

#%%
# Plot angular power spectra

#%%

plt.plot(ells, D_ells_perp, color="tab:blue", label=r'$D_{\ell, \perp}$')
plt.plot(ells, D_ells_par, color="tab:red", label=r'$D_{\ell, \parallel}$')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="lower left")
#plt.savefig('kSZ_angular_power_spectra.pdf',  format="pdf", bbox_inches="tight")
plt.show()
