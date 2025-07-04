import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from scipy.special import erf

#%%
# Functions to compute the integrals

#%%

def P_perp_1(k, pk_mm, pk_eg, a, aHf):

    mu_vals = np.linspace(-0.99, 0.99, 128)
    lk_vals = np.log(np.logspace(-4, 1, 128))

    def integrand2(mu, kp):
        q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
        return (1-mu**2) * pk_eg(q, a, cosmo)

    def integrand1(lkp):
        kp = np.exp(lkp)
        integrand = integrand2(mu_vals, kp)
        integral = np.trapz(integrand, mu_vals)
        return kp * integral * pk_mm(kp, a, cosmo)

    integrand = np.array([integrand1(lk) for lk in lk_vals])
    integral = np.trapz(integrand, lk_vals)
    
    return integral * aHf**2 / (2*np.pi)**2


def P_perp_2(k, pk_em, pk_gm, a, aHf):

    mu_vals = np.linspace(-0.99, 0.99, 128)
    lk_vals = np.log(np.logspace(-4, 1, 128))
    
    def integrand2(mu, kp):
        q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
        return -(1-mu**2) * pk_gm(q, a) / q**2
    
    def integrand1(lkp):
        kp = np.exp(lkp)
        integrand = integrand2(mu_vals, kp)
        integral = np.trapz(integrand, mu_vals)
        return kp**3 * integral * pk_em(kp, a)

    integrand = np.array([integrand1(lk) for lk in lk_vals])
    integral = np.trapz(integrand, lk_vals)
    
    return integral * aHf**2 / (2*np.pi)**2
   

def P_par_1(k, pk_mm, pk_eg, a, aHf):

    mu_vals = np.linspace(-0.99, 0.99, 128)
    lk_vals = np.log(np.logspace(-4, 1, 128))

    def integrand2(mu, kp):
        q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
        return mu**2 * pk_eg(q, a)

    def integrand1(lkp):
        kp = np.exp(lkp)
        integrand = integrand2(mu_vals, kp)
        integral = np.trapz(integrand, mu_vals)
        return kp * integral * pk_mm(kp, a)

    integrand = np.array([integrand1(lk) for lk in lk_vals])
    integral = np.trapz(integrand, lk_vals)
    
    return integral * aHf**2 / (2*np.pi)**2


def P_par_2(k, pk_em, pk_gm, a, aHf):

    mu_vals = np.linspace(-0.99, 0.99, 128)
    lk_vals = np.log(np.logspace(-4, 1, 128))
    
    def integrand2(mu, kp):
        q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
        return mu * (k-kp*mu) * pk_gm(q, a) / q**2
    
    def integrand1(lkp):
        kp = np.exp(lkp)
        integrand = integrand2(mu_vals, kp)
        integral = np.trapz(integrand, mu_vals)
        return kp**2 * integral * pk_em(kp, a)

    integrand = np.array([integrand1(lk) for lk in lk_vals])
    integral = np.trapz(integrand, lk_vals)
    
    return integral * aHf**2 / (2*np.pi)**2


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
lk_arr = np.log(k_vals)
a_arr = np.linspace(0.1, 1, 32)

log10M = np.linspace(10, 15, 1000)
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

# Function to normalise into an overdensity
def p_gas_normalisation(pgas, a):
    '''
    Calculates the physical gas density at scale factor a for a given gas profile p_gas
    
    '''
    def rho_gas_integrand(M):
        fb, fe, fs = pgas._get_fractions(cosmo, M)
        return (fb + fe) * M * pgas.prefac_rho
    
    return hmc.integrate_over_massfunc(rho_gas_integrand, cosmo, a) / a**3

# Normalisation factor
gas_norm = p_gas_normalisation(pGas, a)

#%%
# Cross-correlations

#%%

# Matter-matter
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-matter
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter
pk_em = (1/gas_norm) * ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy
pk_eg = (1/gas_norm) * ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy one-halo term only
pk_eg_1h = (1/gas_norm) * ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_2h=False)

# Electron-electron
pk_ee = (1/gas_norm)**2 * ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pGas, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-galaxy
pk_gg = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Plot the different contributions
plt.plot(k_vals, pk_mm(k_vals, a), label=r'm-m', color="tab:blue")
plt.plot(k_vals, pk_gm(k_vals, a), label=r'g-m', color="tab:red")
plt.plot(k_vals, pk_em(k_vals, a), label=r'e-m', color="tab:cyan")
plt.plot(k_vals, pk_eg(k_vals, a), label=r'e-g', color="tab:purple")
plt.plot(k_vals, pk_ee(k_vals, a), label=r'e-e', color="tab:pink")
plt.plot(k_vals, pk_gg(k_vals, a), label=r'g-g', color="tab:green")
plt.loglog()
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P(k)$', fontsize=20)
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectra_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# 3D power spectra calculations

#%%

# Perpendicular mode
P_of_k_perp_1 = np.array([P_perp_1(k, pk_mm, pk_eg, a, aHf) for k in k_vals])
P_of_k_perp_2 = np.array([P_perp_2(k, pk_em, pk_gm, a, aHf) for k in k_vals])
P_of_k_perp_T = P_of_k_perp_1 + P_of_k_perp_2

# Parallel mode
P_of_k_par_1 = np.array([P_par_1(k, pk_mm, pk_eg, a, aHf) for k in k_vals])
P_of_k_par_2 = np.array([P_par_2(k, pk_em, pk_gm, a, aHf) for k in k_vals])
P_mm = aHf**2 * pk_mm(k_vals, a) / ((2 * np.pi)**3 * k_vals**2)
P_of_k_par_T = P_of_k_par_1 + P_of_k_par_2 # + P_mm

#%%
# Plot 3D power spectra

#%%

plt.plot(k_vals, P_of_k_perp_T, label=r'$P_{q_\perp,1} + P_{q_\perp,2}$', color='tab:red')
plt.plot(k_vals, P_of_k_perp_1, label=r'$P_{q_\perp,1}$', color='tab:blue', linestyle='--')
plt.plot(k_vals, -P_of_k_perp_2, label=r'$-P_{q_\perp,2}$', color='tab:cyan', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_vals, P_of_k_par_T, label=r'$P_{q_\parallel,1} + P_{q_\parallel,2}$', color='tab:red')
plt.plot(k_vals, P_of_k_par_1, label=r'$P_{q_\parallel,1}$', color='tab:blue', linestyle='--')
plt.plot(k_vals, P_of_k_par_2, label=r'$P_{q_\parallel,2}$', color='tab:cyan', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Impact of the two-halo term

#%%

plt.plot(k_vals, pk_eg(k_vals, a), label=r'$P_{\rm eg}^{\rm 1h} + P_{\rm eg}^{\rm 2h}$', color="tab:blue")
plt.plot(k_vals, pk_eg_1h(k_vals, a), label=r'$P_{\rm eg}^{\rm 1h}$', color="tab:red")
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{\rm eg}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_eg_two_halo_term.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Impact of satellite galaxies

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
    return np.trapz(integrand_1, log10_M) / np.trapz(integrand_2, log10_M)
    
#%%

# Central galaxy only calculation
n_M = nM(cosmo, M, a)
M0_cen = 1e15
M1_cen = 1e13
M_min_cen = 3.14e12 # value of M_min_cen at which M_mean = 1e13
N_cen_1 = N_c(M, M_min_cen)
N_sat_1 = N_s(M, M0_cen, M1_cen)
M_mean_cen = mean_halo_mass(M, n_M, N_cen_1, N_sat_1)
print(f"{M_mean_cen:.2e}")

# Central and satellites calculation
M0_sat = 1e11
M1_sat = 3e12
M_min_sat = 1.59e11 # value of M_min_sat at which M_mean = 1e13
N_cen_2 = N_c(M, M_min_sat)
N_sat_2 = N_s(M, M0_sat, M1_sat)
M_mean_sat = mean_halo_mass(M, n_M, N_cen_2, N_sat_2)
print(f"{M_mean_sat:.2e}")

# HOD halo profiles
pg_cen = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(M_min_cen), log10M0_0=np.log10(M0_cen), log10M1_0=np.log10(M1_cen), alpha_0=1.2)
pg_sat = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(M_min_sat), log10M0_0=np.log10(M0_sat), log10M1_0=np.log10(M1_sat), alpha_0=1.2)

#%%

# Electron-galaxy power spectrum with central galaxy only
pk_eg_cen = (1/gas_norm) * ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pg_cen, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy power spectrum with central and satellite galaxies
pk_eg_sat = (1/gas_norm) * ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pg_sat, lk_arr=lk_arr, a_arr=a_arr)

#%%

plt.plot(k_vals, pk_eg_cen(k_vals, a), label=r'$P_{\rm eg}$ with central only', color="tab:red")
plt.plot(k_vals, pk_eg_sat(k_vals, a), label=r'$P_{\rm eg}$ with central + satellites', color="tab:blue")
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{\rm eg}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False, loc='lower left')
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_satellite_galaxies.pdf', format="pdf", bbox_inches="tight")
plt.show()
