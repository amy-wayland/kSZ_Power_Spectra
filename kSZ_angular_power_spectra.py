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
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

k_vals = np.logspace(-3, 1, 128)
k_prime_vals = np.logspace(-3, 1, 128)
lk_arr = np.log(k_vals)
a_arr = np.linspace(0.1, 1, 16)

H = cosmo['h']/ccl.h_over_h0(cosmo, a_arr)
a_dot = a_arr * H

D = ccl.growth_factor(cosmo, a_arr)
ln_a = np.log(a_arr)
ln_D = np.log(D)
f = np.gradient(ln_D, ln_a)

#%%
# Define functions to compute the integrals for the first two contributions to P_{q_\perp}

#%%

def double_integral_t1(k, k_prime_vals, P_of_k_1, P_of_k_2, a_index, a):
    '''
    Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term

    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_m \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_e}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    a_dot_val = a * H[a_index] / (100 * cosmo["h"])
    f_val = f[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 2000)

    def integrand(mu, k_prime):
        q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
        return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * (1 - mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


def double_integral_t2(k, k_prime_vals, P_of_k_1, P_of_k_2, a_index, a):
    '''
    Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term

    '''
    a_dot_val = a * H[a_index] / (100 * cosmo["h"])
    f_val = f[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 2000)
    
    def integrand(mu, k_prime):
        p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        q = np.sqrt(p)
        return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * k_prime**2 * (1-mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
    
    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


#%%
# Define functions to compute the integrals for the first two contributions to P_{q_\parallel}

#%%

def double_integral_t1_par(k, k_prime_vals, P_of_k_1, P_of_k_2, a_index, a):
    '''
    Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term

    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_m \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_e}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    a_dot_val = a * H[a_index] / (100 * cosmo["h"])
    f_val = f[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 2000)

    def integrand(mu, k_prime):
        q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
        return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * mu**2 * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals) / k**2


def double_integral_t2_par(k, k_prime_vals, P_of_k_1, P_of_k_2, a_index, a):
    '''
    Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term

    '''
    a_dot_val = a * H[a_index] / (100 * cosmo["h"])
    f_val = f[a_index]
    mu_vals = np.linspace(-0.99, 0.99, 2000)
    
    def integrand(mu, k_prime):
        #p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        p = np.maximum(k**2 + k_prime**2 - 2 * k * k_prime * mu, 1e-4)
        q = np.sqrt(p)
        return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * mu * (k/k_prime - mu) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
    
    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals) / k**2


#%%
# Halo model implementation

#%%

# We will use a mass definition with Delta = 200 times the matter density
hmd_200m = ccl.halos.MassDef200m

# The Duffy 2008 concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

# The Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)

# The Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

# The NFW profile to characterise the matter density around halos
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

# The HOD model to characterise galaxy overdensity
pg = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)

# Create a HMCalculator object for the mass integrals
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=10., nM=32)

#%%

# Normalisation method for kind="rho_gas"
profile_parameters = {"lMc": 14.0, "beta": 1.0, "eta_b": 0.05}
profile_density = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
profile_density.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)

# Contribution from baryons
rho_crit = ccl.physical_constants.RHO_CRITICAL
rho_bar = cosmo["Omega_b"] * rho_crit

# Contribution from stars
a = 1/(1+0.55)
log10M = np.linspace(11, 15, 1000)
M = 10**log10M
n_M = nM(cosmo, M, a)
f_bound, f_ejected, f_star = profile_density._get_fractions(cosmo, M)
integrand = M * n_M * f_star
rho_star = np.trapz(integrand, log10M) / a**3

# Normalisation factor to convert the electron density into the electron overdensity
rho_mean = rho_bar + rho_star

#%%

# Matter-matter power spectrum
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=np.log(k_vals), a_arr=a_arr)

# Galaxy-matter power spectrum
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter power spectrum
pk_em = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy power spectrum
pk_eg = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg, lk_arr=lk_arr, a_arr=a_arr) # with two-halo term

#%%
# Now calculate the 3D power spectra for the transverse and longitudinal components

#%%

P_of_k_2d_par = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_par = np.array([double_integral_t1_par(k, k_prime_vals, pk_mm, pk_eg, i, a) for k in k_vals])
    #P2_par = np.array([double_integral_t2_par(k, k_prime_vals, pk_em, pk_gm, i, a) for k in k_vals])
    P_of_k_2d_par[:, i] = P1_par

    P1 = np.array([double_integral_t1(k, k_prime_vals, pk_mm, pk_eg, i, a) for k in k_vals])
    #P2 = np.array([double_integral_t2(k, k_prime_vals, pk_em, pk_gm, i, a) for k in k_vals])
    P_of_k_2d[:, i] = P1

z = (1 / a_arr) - 1
sorted_indices = np.argsort(a_arr)

z = z[sorted_indices]
a_arr = a_arr[sorted_indices]

P_of_k_2d = P_of_k_2d[:, sorted_indices]
P_of_k_2d_par = P_of_k_2d_par[:, sorted_indices]

pk2d = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d, is_logp=False)
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
sorted_indices = np.argsort(a_arr)
z = z[sorted_indices]
a_arr = a_arr[sorted_indices]

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

ells = np.geomspace(2, 1000, 20)
C_ells = sigma_T * n_e0 * ((ells * (ells+1)) / (ells+1/2)**2) * ccl.angular_cl(cosmo, gc_pi, gc_T, ells, p_of_k_a=pk2d) / 2
C_ells_par = sigma_T * n_e0 * ((ells * (ells+1)) / (ells+1/2)**2) * ccl.angular_cl(cosmo, gc_pi_par, gc_T_par, ells, p_of_k_a=pk2d_par) / 4

D_ells = ells * (ells + 1) * C_ells / (2 * np.pi)
D_ells_par = ells * (ells + 1) * C_ells_par / (2 * np.pi)

print(C_ells)
print(C_ells_par)

#%%
# Plot

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

plt.plot(ells, D_ells, color="tab:blue", label=r'$D_{\ell, \perp}$')
plt.plot(ells, D_ells_par, color="tab:red", label=r'$D_{\ell, \parallel}$')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell (\ell + 1) \, C_{\ell}^{\pi T} / (2\pi)$', fontsize=16)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=18, frameon=False, loc="lower left")
#plt.savefig('kSZ_angular_power_spectra_longitduinal_vs_transverse.pdf',  format="pdf", bbox_inches="tight")
plt.show()