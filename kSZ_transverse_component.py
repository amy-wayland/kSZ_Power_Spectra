import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from scipy.integrate import simps

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

# Define the k and a values
k_vals = np.logspace(-3, -1, 30)
k_prime_vals = np.logspace(-3, 1, 256)
lk_arr = np.log(k_vals)
a_arr = np.linspace(0.1, 1, 32)

# Scale factor and associated quantities
a = 1/(1+0.55)
H = cosmo['h']/ccl.h_over_h0(cosmo, a)
a_dot = a * H
D = ccl.growth_factor(cosmo, a)
ln_a = np.log(a)
ln_D = np.log(D)
f = ln_D / ln_a # np.gradient(ln_D, ln_a)

#%%
# Define functions to compute the integrals for the first two contributions to P_{q_\perp}

#%%

def double_integral_t1(k, k_prime_vals, P_of_k_1, P_of_k_2):
    '''
    Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term

    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_m \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_e}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    mu_vals = np.linspace(-0.99, 0.99, 1000)

    def integrand(mu, k_prime):
        q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
        return (a_dot * f)**2 * (1/(2 * np.pi)**2) * (1 - mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


def double_integral_t2(k, k_prime_vals, P_of_k_1, P_of_k_2):
    '''
    Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term

    '''
    mu_vals = np.linspace(-0.99, 0.99, 1000)
    
    def integrand(mu, k_prime):
        p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        q = np.sqrt(p)
        return (a_dot * f)**2 * (1/(2 * np.pi)**2) * k_prime**2 * (1-mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / p
    
    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)
   
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
hmc.precision['padding_lo_fftlog'] = 1e-2
hmc.precision['padding_hi_fftlog'] = 1e2
hmc.precision['n_per_decade'] = 2000

#%%

# Normalisation method for kind="n_electron"
profile_parameters = {"lMc": 14.0, "beta": 1.0, "eta_b": 0.05}
profile_density = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="n_electron", **profile_parameters)

from astropy.constants import m_p
from astropy.constants import M_sun
from astropy import units as u

rho_crit_cgs = ccl.physical_constants.RHO_CRITICAL * M_sun.to("g") / (u.Mpc.to("cm"))**3  # g/cm^3
rho_b_cgs = cosmo["Omega_b"] * rho_crit_cgs  # g/cm^3

mu_e = 1.14
n_e_bar = (rho_b_cgs / (mu_e * m_p.cgs.value)).value  # in cm^-3
rho_mean = n_e_bar

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
log10M = np.linspace(10, 15, 1000)
M = 10**log10M
n_M = nM(cosmo, M, a)
f_bound, f_ejected, f_star = profile_density._get_fractions(cosmo, M)
integrand = M * n_M * f_star
rho_star = simps(integrand, log10M) / a**3

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
pk_eg_1h = (1/rho_mean) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=pg, lk_arr=lk_arr, a_arr=a_arr, get_2h=False) # one-halo term only

# Electron-electron power spectrum
pk_ee = (1/rho_mean**2) * ccl.halos.halomod_Pk2D(cosmo, hmc, profile_density, prof2=profile_density, lk_arr=lk_arr, a_arr=a_arr)

# Matter-matter power spectrum calculated via HaloProfiles.py
profile_baryon = hp.HaloProfileNFWBaryon(mass_def=hmd_200m, concentration=cM, **profile_parameters)
pk_mm_nfw_bar = ccl.halos.halomod_Pk2D(cosmo, hmc, profile_baryon, lk_arr=lk_arr, a_arr=a_arr)

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

# Test plots of the different contributions
plt.plot(k_vals, pk_mm(k_vals, 1.0), label=r'm-m', color="tab:blue")
plt.plot(k_vals, pk_gm(k_vals, 1.0), label=r'g-m', color="tab:red")
plt.plot(k_vals, pk_em(k_vals, 1.0), label=r'e-m', color="tab:cyan")
plt.plot(k_vals, pk_eg(k_vals, 1.0), label=r'e-g', color="tab:pink")
plt.plot(k_vals, pk_ee(k_vals, 1.0), label=r'e-e', color="tab:green")
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectra_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Now perform the calculations

#%%

# Calculate the integral for different values of k
P_of_k_term_1 = np.array([double_integral_t1(k, k_prime_vals, pk_mm, pk_eg) for k in k_vals])
P_of_k_term_2 = np.array([double_integral_t2(k, k_prime_vals, pk_em, pk_gm) for k in k_vals])

# For the electron-galaxy one-halo term
P_of_k_term_1_1h = np.array([double_integral_t1(k, k_prime_vals, pk_mm, pk_eg_1h) for k in k_vals])

# Connected non-Gaussian contribution
P_4h_vals = [np.float64(2170.16363495884), np.float64(2204.4672372778305), np.float64(2195.320290446668), np.float64(2186.980642412834), np.float64(2179.3428931544154), np.float64(2172.069973997958), np.float64(2165.0590526063565), np.float64(2158.378004428503), np.float64(2149.076272129585), np.float64(2138.7398007596853), np.float64(2125.205111523403), np.float64(2110.3170948344655), np.float64(2093.8137114754413), np.float64(2075.427725294425), np.float64(2054.8886843159094), np.float64(2032.1156301724861), np.float64(2006.9617851090381), np.float64(1980.688285945524), np.float64(1953.590734928418), np.float64(1924.4092312013945), np.float64(1893.2618926661157), np.float64(1860.5667263519492), np.float64(1826.5378552595905), np.float64(1791.1779632594212), np.float64(1748.1628446965185), np.float64(1704.2391478822276), np.float64(1661.884231504758), np.float64(1619.0978290226446), np.float64(1575.1685964258907), np.float64(1530.808789318616)]

#%%
# Plot the results

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

plt.plot(k_vals, P_of_k_term_1 - P_of_k_term_2, label=r'$P_{q_\perp,1} + P_{q_\perp,2}$', color='tab:red')
plt.plot(k_vals, P_of_k_term_1, label=r'$P_{q_\perp,1}$', color='tab:blue', linestyle='--')
plt.plot(k_vals, P_of_k_term_2, label=r'$-P_{q_\perp,2}$', color='tab:cyan', linestyle='--')
plt.plot(k_vals, P_4h_vals)
plt.xlim(1e-3, 1e-1)
plt.ylim(1e1, 1e6)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_contributions.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

plt.plot(k_vals, pk_eg(k_vals, 1.0), label=r'one-halo + two-halo', color="tab:blue")
plt.plot(k_vals, pk_eg_1h(k_vals, 1.0), label=r'one-halo only', color="tab:red")
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{\rm eg}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_eg_two_halo_term.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

plt.plot(k_vals, P_of_k_term_1, label=r'$P_{q_\perp,1}$ with one-halo and two-halo term', color='tab:blue', linestyle='--')
plt.plot(k_vals, P_of_k_term_1_1h, label=r'$P_{q_\perp,1}$ with one-halo term only', color='tab:green', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_first_term.pdf', format="pdf", bbox_inches="tight")
plt.show()