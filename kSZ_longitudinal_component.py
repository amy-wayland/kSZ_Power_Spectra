import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
from scipy.integrate import simps
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
             #"transfer_function": "eisenstein_hu"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

# Define the k and a values
k_vals = np.logspace(-3, 1, 256)
k_prime_vals = np.logspace(-3, 1, 256)
lk_arr = np.log(k_vals)
a_arr = np.linspace(0.1, 1, 32)

a = 1/(1+0.55)
H = cosmo['h']/ccl.h_over_h0(cosmo, a)
f = ccl.growth_rate(cosmo, a)
a_dot = a * H

#%%
# Define functions to compute the integrals for the first two contributions to P_{q_\parallel}

#%%

def double_integral_t1_par(k, k_prime_vals, P_of_k_1, P_of_k_2):
    '''
    Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term

    Parameters
    ----------
    k : wavenumber k
    k_prime_vals : values of k' over which to integrate
    P_of_k_1 : P_{\delta_m \delta_m}(k')
    P_of_k_2 : P_{\delta_g \delta_e}(\sqrt{k^2 + (k')^2 - 2 k k' \mu})

    '''
    mu_vals = np.linspace(-0.99, 0.99, 2000)

    def integrand(mu, k_prime):
        q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
        return (a_dot * f)**2 * (1/(2 * np.pi)**2) * mu**2 * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)


def double_integral_t2_par(k, k_prime_vals, P_of_k_1, P_of_k_2):
    '''
    Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term

    '''
    mu_vals = np.linspace(-0.99, 0.99, 2000)
    
    def integrand(mu, k_prime):
        #p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        p = np.maximum(k**2 + k_prime**2 - 2 * k * k_prime * mu, 1e-4)
        q = np.sqrt(p)
        return (a_dot * f)**2 * (1/(2 * np.pi)**2) * (mu * k_prime * k) * (1 - (k_prime / k) * mu) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
    
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

#%%

# Normalisation method for kind="rho_gas"
profile_parameters = {"lMc": 14.0, "beta": 1.0, "eta_b": 0.05}
profile_density = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
profile_density.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=2000, plaw_fourier=-2.0)

# Contribution from baryons
rho_crit = ccl.physical_constants.RHO_CRITICAL
rho_bar = cosmo["Omega_b"] * rho_crit

# Contribution from stars
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
# Now perform the calculations

#%%

# Calculate the integral for different values of k
P_of_k_term_1_par = np.array([double_integral_t1_par(k, k_prime_vals, pk_mm, pk_eg) for k in k_vals])
P_of_k_term_2_par = np.array([double_integral_t2_par(k, k_prime_vals, pk_em, pk_gm) for k in k_vals])

# For the one-halo term only
#P_of_k_term_1_1h_par = np.array([double_integral_t1_par(k, k_prime_vals, pk_mm, pk_eg_1h) for k in k_vals])

#%%
# Compare the transverse and longitudinal terms

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
    mu_vals = np.linspace(-0.99, 0.99, 2000)

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
    mu_vals = np.linspace(-0.99, 0.99, 2000)
    
    def integrand(mu, k_prime):
        p = k**2 + k_prime**2 - 2 * k * k_prime * mu
        q = np.sqrt(p)
        return (a_dot * f)**2 * (1/(2 * np.pi)**2) * k_prime**2 * (1-mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
    
    def int_over_mu(k_prime):
        vals = integrand(mu_vals, k_prime)
        return np.trapz(vals, mu_vals)

    integrand_k_prime = np.array([int_over_mu(k_p) for k_p in k_prime_vals])
    
    return np.trapz(integrand_k_prime, k_prime_vals)

#%%

# Calculate the integral for different values of k
P_of_k_term_1 = np.array([double_integral_t1(k, k_prime_vals, pk_mm, pk_eg) for k in k_vals])
P_of_k_term_2 = np.array([double_integral_t2(k, k_prime_vals, pk_em, pk_gm) for k in k_vals])

#%%

# For the electron-galaxy one-halo term
P_of_k_term_1_1h = np.array([double_integral_t1(k, k_prime_vals, pk_mm, pk_eg_1h) for k in k_vals])

#%%

# Contributions from the connected non-Gaussian term
P_4h_vals_par = [np.float64(14.45642728795), np.float64(22.767442797707243), np.float64(31.666531176779255), np.float64(39.52552906746157), np.float64(46.770012672227686), np.float64(53.461686623460686), np.float64(60.13451205621642), np.float64(66.06334375397579), np.float64(70.74123047943358), np.float64(76.67764476391633), np.float64(82.32430588962895), np.float64(87.99066538654822), np.float64(93.58022062269688), np.float64(99.40052361774954), np.float64(105.379564391522), np.float64(111.65189155044263), np.float64(118.87753351105609), np.float64(128.4413623116191), np.float64(138.40547414147326), np.float64(149.4478314206271), np.float64(162.33281364227366), np.float64(177.71952422494198), np.float64(196.47784160088418), np.float64(218.50389951990206), np.float64(244.70257075795064), np.float64(275.9938567408744), np.float64(311.4923429405752), np.float64(351.0288283767405), np.float64(395.4327258824649), np.float64(445.5366524497683)]
P_1h_vals_par = [np.float64(6.252171790965269e-11), np.float64(7.090167235522679e-11), np.float64(8.008630496283613e-11), np.float64(8.995315614251391e-11), np.float64(1.0032549399788202e-10), np.float64(1.1122273017633746e-10), np.float64(1.2279324429470357e-10), np.float64(1.3507201205918666e-10), np.float64(1.4804002872799613e-10), np.float64(1.6161907062647286e-10), np.float64(1.7566936531300606e-10), np.float64(1.8998109592161927e-10), np.float64(2.0427147256229123e-10), np.float64(2.181762894016502e-10), np.float64(2.3124488867946125e-10), np.float64(2.4279816936802227e-10), np.float64(2.5171616624425264e-10), np.float64(2.5697925558085607e-10), np.float64(2.591342951525006e-10), np.float64(2.5919064460324953e-10), np.float64(2.577345624543855e-10), np.float64(2.5516310806108495e-10), np.float64(2.5178709356955186e-10), np.float64(2.478449666807619e-10), np.float64(2.435363493768045e-10), np.float64(2.3909075742817447e-10), np.float64(2.347456525346597e-10), np.float64(2.3030733996458552e-10), np.float64(2.2480162832512217e-10), np.float64(2.162004520569935e-10)]

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

plt.plot(k_vals, P_of_k_term_1, label=r'$P_{q_{\perp}}^{\pi T}$', color='tab:blue')
plt.plot(k_vals, P_of_k_term_1_par, label=r'$P_{q_{\parallel}}^{\pi T}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=18, frameon=False, loc="lower left")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_transverse_vs_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12})

P_mm = ((a_dot * f)**2) * pk_mm(k_vals, 1/(1+0.55)) / ((2 * np.pi)**3 * k_vals**2)

plt.plot(k_vals, P_of_k_term_1_par, label=r'$P_{q_{\parallel,1}}^{\pi T}$', color='tab:blue')
plt.plot(k_vals, P_of_k_term_2_par, label=r'$P_{q_{\parallel,2}}^{\pi T}$', color='tab:red')
plt.plot(k_vals, P_mm, label=r'$P_{\rm mm}$', color='tab:purple')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P^{\pi T}_{q_\parallel}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=18, frameon=False, loc="lower left")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_longitudinal_contributions.pdf', format="pdf", bbox_inches="tight")
plt.show()
