import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt

#%%
# Functions to compute the integrals

#%%

def P_perp_1(k, pk_mm, pk_eg, a, aHf_arr, a_index):
    
    aHf = aHf_arr[a_index]
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


def P_perp_2(k, pk_em, pk_gm, a, aHf_arr, a_index):

    aHf = aHf_arr[a_index]
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
   

def P_par_1(k, pk_mm, pk_eg, a, aHf_arr, a_index):

    aHf = aHf_arr[a_index]
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


def P_par_2(k, pk_em, pk_gm, a, aHf_arr, a_index):

    aHf = aHf_arr[a_index]
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
pG = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=12.89, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1, bg_0=2.04)

# Halo model integral calculator
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m, log10M_max=15., log10M_min=10., nM=32)

# Gas density profile
profile_parameters = {"lMc": 14.0, "beta": 0.6, "eta_b": 0.5, "A_star": 0.03}
pGas = hp.HaloProfileDensityHE(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
pGas.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=300, plaw_fourier=-2.0)

def p_gas_normalisation(pgas, a):
    '''
    Calculates the physical gas density at scale factor a for a given gas profile p_gas
    to normalise the gas density profile
    
    '''
    def rho_gas_integrand(M):
        fb, fe, fs = pgas._get_fractions(cosmo, M)
        return (fb + fe) * M * pgas.prefac_rho
    
    return hmc.integrate_over_massfunc(rho_gas_integrand, cosmo, a) / a**3


def pk_xe(cosmo, hmc, prof, *,
                 prof2=None, prof_2pt=None,
                 p_of_k_a=None,
                 get_1h=True, get_2h=True,
                 lk_arr=None, a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2,
                 smooth_transition=None, suppress_1h=None, extrap_pk=False):
    '''
    Returns a Pk2D object of the cross-correlation between the electron overdensity
    with another overdensity profile x correctly normalised

    '''
    pk_arr = ccl.halos.pk_2pt.halomod_power_spectrum(
      cosmo, hmc, np.exp(lk_arr), a_arr,
      prof, prof2=prof2, prof_2pt=prof_2pt, p_of_k_a=p_of_k_a,
      get_1h=get_1h, get_2h=get_2h,
      smooth_transition=smooth_transition, suppress_1h=suppress_1h,
      extrap_pk=extrap_pk)
    
    if isinstance(prof2, hp.HaloProfileDensityHE):
        pk_norm = np.array([p_gas_normalisation(prof, a)**2 for a in a_arr])
    else:
        pk_norm = np.array([p_gas_normalisation(prof, a) for a in a_arr])
    
    pk_arr_normalised = pk_arr / pk_norm[:, None]

    return ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr_normalised,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                is_logp=False)

#%%
# Cross-correlations

#%%

# Matter-matter
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-matter
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter
pk_em = pk_xe(cosmo, hmc, pGas, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy
pk_eg = pk_xe(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy one-halo term only
pk_eg_1h = pk_xe(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_2h=False)

# Electron-electron
pk_ee = pk_xe(cosmo, hmc, pGas, prof2=pGas, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-galaxy
pk_gg = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

#%%
# 3D power spectra calculations for a single a value

#%%

z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = np.array([a * H * f])
a_index = 0

P1_perp = np.array([P_perp_1(k, pk_mm, pk_eg, a, aHf, a_index) for k in k_vals])
P2_perp = np.array([P_perp_2(k, pk_em, pk_gm, a, aHf, a_index) for k in k_vals])
P_perp_T = P1_perp + P2_perp

P1_par = np.array([P_par_1(k, pk_mm, pk_eg, a, aHf, a_index) for k in k_vals])
P2_par = np.array([P_par_2(k, pk_em, pk_gm, a, aHf, a_index) for k in k_vals])
P_par_T = P1_par + P2_par

#%%

plt.plot(k_vals, P_perp_T, label=r'$P_{q_\perp,1} + P_{q_\perp,2}$', color='tab:red')
plt.plot(k_vals, P1_perp, label=r'$P_{q_\perp,1}$', color='tab:blue', linestyle='--')
plt.plot(k_vals, -P2_perp, label=r'$-P_{q_\perp,2}$', color='tab:cyan', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_vals, P_par_T, label=r'$P_{q_\parallel,1} + P_{q_\parallel,2}$', color='tab:red')
plt.plot(k_vals, P1_par, label=r'$P_{q_\parallel,1}$', color='tab:blue', linestyle='--')
plt.plot(k_vals, P2_par, label=r'$P_{q_\parallel,2}$', color='tab:cyan', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# 3D power spectra calculations for an array of a values

#%%

# Perpendicular component

P_of_k_2d_perp_1 = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_perp_2 = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_perp = np.array([P_perp_1(k, pk_mm, pk_eg, a, aHf_arr, i) for k in k_vals])
    P2_perp = np.array([P_perp_2(k, pk_em, pk_gm, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_perp_1[:, i] = P1_perp
    P_of_k_2d_perp_2[:, i] = P2_perp
    
#%%

# Parallel component

P_of_k_2d_par_1 = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_par_2 = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_par = np.array([P_par_1(k, pk_mm, pk_eg, a, aHf_arr, i) for k in k_vals])
    P2_par = np.array([P_par_2(k, pk_em, pk_gm, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_par_1[:, i] = P1_par
    P_of_k_2d_par_2[:, i] = P2_par
    
#%%

z = (1 / a_arr) - 1
sorted_indices = np.argsort(a_arr)
z = z[sorted_indices]
a_arr = a_arr[sorted_indices]
H_arr = H_arr[sorted_indices]
f_arr = f_arr[sorted_indices]

#%%

P_of_k_2d_perp_1 = P_of_k_2d_perp_1[:, sorted_indices]
P_of_k_2d_perp_2 = P_of_k_2d_perp_2[:, sorted_indices]

pk2d_perp_1 = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp_1.T, is_logp=False)
pk2d_perp_2 = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp_2.T, is_logp=False)

#%%

P_of_k_2d_par_1 = P_of_k_2d_par_1[:, sorted_indices]
P_of_k_2d_par_2 = P_of_k_2d_par_2[:, sorted_indices]

pk2d_par_1 = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par_1.T, is_logp=False)
pk2d_par_2 = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par_2.T, is_logp=False)

#%%
# Create custom tracers to calculate the angular power spectrum

#%%

ells = np.geomspace(40, 7979, 7940)
perp_prefac = 0.5 * ells * (ells+1) / (ells+0.5)**2

sigma_T_cgs = 6.65e-25 # cm^2
n_e0_cgs = 8e-4 # cm^{-3}
cm_per_Mpc = 3.0857e24 # cm / Mpc
sigma_T = (sigma_T_cgs / cm_per_Mpc**2) * cosmo['h']**2 # (Mpc/h)**2
n_e0 = (n_e0_cgs * cm_per_Mpc**3) / cosmo['h']**3 # 1/(Mpc/h)**3
A = n_e0 * sigma_T

nz = (1 / np.sqrt(2 * np.pi * 0.05**2)) * np.exp(- 0.5 * ((z - 0.55) / 0.05)**2)

sort_idx = np.argsort(z)
z = z[sort_idx]
nz = nz[sort_idx]

kernel_g = ccl.get_density_kernel(cosmo, dndz=(z,nz))

chis = ccl.comoving_radial_distance(cosmo, 1/(1+z))

tk_perp = ccl.Tracer()
tg_perp = ccl.Tracer()

tk_perp.add_tracer(cosmo, kernel=(chis, 1/a_arr**2))
tg_perp.add_tracer(cosmo, kernel=kernel_g)

tk_par = ccl.Tracer()
tg_par = ccl.Tracer()

tk_par.add_tracer(cosmo, kernel=(chis, 1/a_arr**2), der_bessel=1)
tg_par.add_tracer(cosmo, kernel=kernel_g, der_bessel=1)

#%%
# Calculate the angular power spectra

#%%

C_ells_perp_1 = A * perp_prefac * ccl.angular_cl(cosmo, tg_perp, tk_perp, ells, p_of_k_a=pk2d_perp_1)
C_ells_perp_2 = A * perp_prefac * ccl.angular_cl(cosmo,tg_perp, tk_perp, ells, p_of_k_a=pk2d_perp_2)
C_ells_perp_T = C_ells_perp_1 + C_ells_perp_2

D_ells_perp_1 = ells * (ells + 1) * C_ells_perp_1 / (2 * np.pi)
D_ells_perp_2 = ells * (ells + 1) * C_ells_perp_2 / (2 * np.pi)
D_ells_perp_T = D_ells_perp_1 + D_ells_perp_2

C_ells_par_1 = -A * ccl.angular_cl(cosmo, tg_par, tk_par, ells, p_of_k_a=pk2d_par_1)
C_ells_par_2 = -A * ccl.angular_cl(cosmo, tg_par, tk_par, ells, p_of_k_a=pk2d_par_2)
C_ells_par_T = C_ells_par_1 + C_ells_par_2

D_ells_par_1 = ells * (ells + 1) * C_ells_par_1 / (2 * np.pi)
D_ells_par_2 = ells * (ells + 1) * C_ells_par_2 / (2 * np.pi)
D_ells_par_T = D_ells_par_1 + D_ells_par_2

#%%
# Plot angular power spectra

#%%

plt.plot(ells, D_ells_perp_T, color="tab:blue", label=r'$D_{\ell, \perp, T}$')
plt.plot(ells, D_ells_perp_1, color="tab:blue", label=r'$D_{\ell, \perp, 1}$', linestyle='--')
plt.plot(ells, D_ells_perp_2, color="tab:blue", label=r'$-D_{\ell, \perp, 2}$', linestyle='dotted')
plt.plot(ells, D_ells_par_T, color="tab:red", label=r'$D_{\ell, \parallel, T}$')
plt.plot(ells, D_ells_par_1, color="tab:red", label=r'$D_{\ell, \parallel, 1}$', linestyle='--')
plt.plot(ells, D_ells_par_2, color="tab:red", label=r'$D_{\ell, \parallel, 2}$', linestyle='dotted')
plt.xlim(40, 8e3)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="center right", ncol=2)
#plt.savefig('kSZ_angular_power_spectra.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%
# Covariance matrix calculation

#%%

# pi-pi auto-correlation

P_of_k_2d_perp_1_gg = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_perp_2_gg = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_perp_gg = np.array([P_perp_1(k, pk_mm, pk_gg, a, aHf_arr, i) for k in k_vals])
    P2_perp_gg = np.array([P_perp_2(k, pk_gm, pk_gm, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_perp_1_gg[:, i] = P1_perp_gg
    P_of_k_2d_perp_2_gg[:, i] = P2_perp_gg

P_of_k_2d_par_1_gg = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_par_2_gg = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_par_gg = np.array([P_par_1(k, pk_mm, pk_gg, a, aHf_arr, i) for k in k_vals])
    P2_par_gg = np.array([P_par_2(k, pk_gm, pk_gm, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_par_1_gg[:, i] = P1_par_gg
    P_of_k_2d_par_2_gg[:, i] = P2_par_gg
    
#%%

# T-T auto-correlation

P_of_k_2d_perp_1_TT = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_perp_2_TT = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_perp_TT = np.array([P_perp_1(k, pk_mm, pk_ee, a, aHf_arr, i) for k in k_vals])
    P2_perp_TT = np.array([P_perp_2(k, pk_em, pk_em, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_perp_1_TT[:, i] = P1_perp_TT
    P_of_k_2d_perp_2_TT[:, i] = P2_perp_TT

P_of_k_2d_par_1_TT = np.zeros((len(k_vals), len(a_arr)))
P_of_k_2d_par_2_TT = np.zeros((len(k_vals), len(a_arr)))

for i, a in enumerate(a_arr):
    P1_par_TT = np.array([P_par_1(k, pk_mm, pk_ee, a, aHf_arr, i) for k in k_vals])
    P2_par_TT = np.array([P_par_2(k, pk_em, pk_em, a, aHf_arr, i) for k in k_vals])
    P_of_k_2d_par_1_TT[:, i] = P1_par_TT
    P_of_k_2d_par_2_TT[:, i] = P2_par_TT
    
#%%
    
z = (1 / a_arr) - 1
sorted_indices = np.argsort(a_arr)
z = z[sorted_indices]
a_arr = a_arr[sorted_indices]
H_arr = H_arr[sorted_indices]
f_arr = f_arr[sorted_indices]

#%%

# Pk2D objects for g-g

P_of_k_2d_perp_1_gg = P_of_k_2d_perp_1_gg[:, sorted_indices]
P_of_k_2d_perp_2_gg = P_of_k_2d_perp_2_gg[:, sorted_indices]

pk2d_perp_1_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp_1_gg.T, is_logp=False)
pk2d_perp_2_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp_2_gg.T, is_logp=False)

P_of_k_2d_par_1_gg = P_of_k_2d_par_1_gg[:, sorted_indices]
P_of_k_2d_par_2_gg = P_of_k_2d_par_2_gg[:, sorted_indices]

pk2d_par_1_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par_1_gg.T, is_logp=False)
pk2d_par_2_gg = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par_2_gg.T, is_logp=False)

#%%

# Pk2D objects for T-T

P_of_k_2d_perp_1_TT = P_of_k_2d_perp_1_TT[:, sorted_indices]
P_of_k_2d_perp_2_TT = P_of_k_2d_perp_2_TT[:, sorted_indices]

pk2d_perp_1_TT = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp_1_TT.T, is_logp=False)
pk2d_perp_2_TT = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_perp_2_TT.T, is_logp=False)

P_of_k_2d_par_1_TT = P_of_k_2d_par_1_TT[:, sorted_indices]
P_of_k_2d_par_2_TT = P_of_k_2d_par_2_TT[:, sorted_indices]

pk2d_par_1_TT = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par_1_TT.T, is_logp=False)
pk2d_par_2_TT = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_vals), pk_arr=P_of_k_2d_par_2_TT.T, is_logp=False)

#%%

# Angular power spectra for C_l^{g g} and C_l^{T T}

C_ells_perp_1_gg = perp_prefac * ccl.angular_cl(cosmo, tg_perp, tg_perp, ells, p_of_k_a=pk2d_perp_1_gg)
C_ells_perp_2_gg = perp_prefac * ccl.angular_cl(cosmo,tg_perp, tg_perp, ells, p_of_k_a=pk2d_perp_2_gg)
C_ells_perp_T_gg = C_ells_perp_1_gg + C_ells_perp_2_gg

C_ells_par_1_gg = ccl.angular_cl(cosmo, tg_par, tg_par, ells, p_of_k_a=pk2d_par_1_gg)
C_ells_par_2_gg = ccl.angular_cl(cosmo,tg_par, tg_par, ells, p_of_k_a=pk2d_par_2_gg)
C_ells_par_T_gg = C_ells_par_1_gg + C_ells_par_2_gg

#%%

# Angular power spectra for C_l^{T T}

C_ells_perp_1_TT = A**2 * perp_prefac * ccl.angular_cl(cosmo, tk_perp, tk_perp, ells, p_of_k_a=pk2d_perp_1_TT)
C_ells_perp_2_TT = A**2 * perp_prefac * ccl.angular_cl(cosmo, tk_perp, tk_perp, ells, p_of_k_a=pk2d_perp_2_TT)
C_ells_perp_T_TT = C_ells_perp_1_TT + C_ells_perp_2_TT

C_ells_par_1_TT = A**2 * ccl.angular_cl(cosmo, tk_par, tk_par, ells, p_of_k_a=pk2d_par_1_TT)
C_ells_par_2_TT = A**2 * ccl.angular_cl(cosmo,tk_par, tk_par, ells, p_of_k_a=pk2d_par_2_TT)
C_ells_par_T_TT = C_ells_par_1_TT + C_ells_par_2_TT

#%%

# Plot the T-T auto-correlation to compare to Ma+Fry 2001

D_ells_perp_TT = -ells * (ells + 1) * C_ells_perp_T_TT / (2 * np.pi)

plt.plot(ells, D_ells_perp_TT)
plt.xlim(40, 8e3)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{T T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="center right", ncol=2)
#plt.savefig('kSZ_angular_power_spectra.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%

# CMB power spectrum

cmb_data = np.loadtxt('data/camb_93159309_scalcls.dat', usecols=(0,1))
ell_vals_full = cmb_data[:,0]
ell_vals_cmb = ell_vals_full[38:7978]
T_cmb = 2.725e6 # micro Kelvin
D_ells_full = cmb_data[:,1]
D_ells_cmb = D_ells_full[38:7978]
C_ells_cmb = np.array([2 * np.pi * D_ells_cmb[i] * 1/(ell_vals_cmb[i] * (ell_vals_cmb[i]+1)) for i in range(len(ell_vals_cmb))]) / T_cmb**2

#%%

# Noise power spectrum for C_l^{TT}

d = np.loadtxt("data/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_CMB.txt", unpack=True)
l_base = d[0]
nl_base = d[1]
l_s4 = l_base

# Atmospheric noise
lknee = 2154
aknee = -3.5

# Noise rms: 2 uK-arcmin
DT = 2.0

# Beam FWHM
fwhm = 1.4
beam = np.exp(-0.5*l_s4*(l_s4+1)*(np.radians(fwhm/60)/2.355)**2)

# White amplitude in uK^2 srad
Nwhite = DT**2*(np.pi/180/60)**2

# S4-only noise
T_cmb = 2.725e6 # micro Kelvin
nl_s4 = (Nwhite * (1 + (l_s4/lknee)**aknee)/beam**2)

# Combine with Planck on large scales
nl_s4 = (1/(1/nl_s4+1/nl_base)) / T_cmb**2

#%%

# Secondary anisotropies power spectrum for C_l^{T T}

import pickle

act_cells = pickle.load(open("data/P-ACT_theory_cells.pkl", "rb"))

ells_act = act_cells["ell"]
ells_act = ells_act[38:7978] # Match the ell values to the noise power spectrum
C_ells_act = act_cells["tt", "dr6_pa5_f090", "dr6_pa5_f090"] / T_cmb**2
C_ells_act = C_ells_act[38:7978]

#%%

# Shot noise power spectrum for C_l^{g g}

sigma_v = (300e3 / 3e8)
n_bar = 149 * (np.pi/180)**2 # galaxies per square radian
nl_pi = sigma_v**2 / n_bar

#%%

ells = np.geomspace(40, 7979, 7940)

f_sky = 0.4
d_ell = (7979 - 40) / 7940

nl_pi_arr = np.full_like(ells, nl_pi)
C_ells_gg = nl_pi_arr + C_ells_perp_T_gg

# Limit 1: optimistic case where we have the CMB and noise only
C_ells_TT_1 = C_ells_cmb + nl_s4
cov_1 = np.sqrt((C_ells_gg * C_ells_TT_1 + C_ells_perp_T**2) / (f_sky * (2 * ells + 1) * d_ell))

# Limit 2: realistic case where we also account for secondary anisotropies
C_ells_TT_2 = C_ells_act + nl_s4
cov_2 = np.sqrt((C_ells_gg * C_ells_TT_2 + C_ells_perp_T**2) / (f_sky * (2 * ells + 1) * d_ell))

print(cov_1)
print(cov_2)

#%%

plt.errorbar(ells, C_ells_perp_T, yerr=cov_1)
plt.xlim(40, 8e3)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$C_{\ell, \perp}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.show()

plt.errorbar(ells, C_ells_perp_T, yerr=cov_2)
plt.xlim(40, 8e3)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$C_{\ell, \perp}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.show()

#%%

plt.plot(ells, C_ells_cmb, label="Primary CMB", color='tab:blue')
plt.plot(ells, C_ells_act, label="ACT secondary", color='tab:red')
plt.plot(ells, nl_s4, label="Noise", color='tab:cyan')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$N_{\ell}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="best")
plt.show()

plt.plot(ells, nl_pi_arr, label="Shot noise", color='tab:blue')
plt.plot(ells, C_ells_perp_T_gg, label="Clustering", color='tab:red')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$N_{\ell}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="best")
plt.show()

sn_squared_1 = np.sum((C_ells_perp_T**2) / (cov_1**2))
print(f"Total S/N (optimistic): {np.sqrt(sn_squared_1):.2f}")

sn_squared_2 = np.sum((C_ells_perp_T**2) / (cov_2**2))
print(f"Total S/N (realistic): {np.sqrt(sn_squared_2):.2f}")

