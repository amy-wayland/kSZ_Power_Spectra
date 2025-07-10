import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import kSZclass as ksz
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

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

k_arr = np.logspace(-3, 1, 128)
lk_arr = np.log(k_arr)
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
pGas = hp.HaloProfileDensityHE_withFT(mass_def=hmd_200m, concentration=cM, kind="rho_gas", **profile_parameters)
pGas.update_precision_fftlog(padding_lo_fftlog=1e-2, padding_hi_fftlog=1e2, n_per_decade=300, plaw_fourier=-2.0)

#%%
# Cross-correlations

#%%

# Matter-matter
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-matter
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-matter
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy
pk_eg = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy one-halo term only
pk_eg_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_2h=False)

# Electron-electron
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pGas, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-galaxy
pk_gg = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

#%%
# Calculate 3D power spectra

#%%

# Initialise a kSZ object using kSZclass.py
kSZ = ksz.kSZclass(cosmo, k_arr, a_arr, pk_mm, pk_eg, pk_gm, pk_em, pk_ee, pk_gg)

z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = np.array([a * H * f])
a_index = 0

# Galaxy-kSZ cross-correlation (transverse)
pkt_gk1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_eg) for k in k_arr])
pkt_gk2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_gm, pk_em) for k in k_arr])
pkt_gk = pkt_gk1 + pkt_gk2

# Galaxy-kSZ cross-correlation (longitudinal)
pkp_gk1 = np.array([kSZ.P_par_1(k, a, a_index, pk_eg) for k in k_arr])
pkp_gk2 = np.array([kSZ.P_par_2(k, a, a_index, pk_gm, pk_em) for k in k_arr])
pkp_gk = pkp_gk1 + pkp_gk2

# Galaxy-galaxy auto-correlation (transverse)
pkt_gg1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_gg) for k in k_arr])
pkt_gg2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_gm, pk_gm) for k in k_arr])
pkt_gg = pkt_gg1 + pkt_gg2

# kSZ-kSZ auto-correlation (transverse)
pkt_kk1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_ee) for k in k_arr])
pkt_kk2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_em, pk_em) for k in k_arr])
pkt_kk = pkt_kk1 + pkt_kk2

# Galaxy-kSZ cross-correlation (transverse) without two-halo term
pkt_gk1_1h = np.array([kSZ.P_perp_1(k, a, a_index, pk_eg_1h) for k in k_arr])
pkt_gk_1h = pkt_gk1_1h + pkt_gk2

# Galaxy-kSZ cross-correlation (transverse) with satellites
log10M = np.linspace(11, 15, 1000)
M_vals = 10**log10M
n_M = nM(cosmo, M_vals, a)
b_M = bM(cosmo, M_vals, a)
sat = ksz.Satellites(M_vals, M0=1e11, M1=3e12, M_min=1.74e12, nM=n_M)
M_mean_sat = sat.mean_halo_mass()

pG_sat = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(1.74e12), log10M0_0=np.log10(1e11), log10M1_0=np.log10(3e12), alpha_0=1.1)
pk_eg_sat = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG_sat, lk_arr=lk_arr, a_arr=a_arr)
pkt_gk1_sat = np.array([kSZ.P_perp_1(k, a, a_index, pk_eg_sat) for k in k_arr])
pkt_gk_sat = pkt_gk1_sat + pkt_gk2

# Compare M_mean to central only case using the HOD parameters from pG
cen = ksz.Satellites(M_vals, M0=10**(12.92), M1=10**(13.95), M_min=10**(12.89), nM=n_M)
M_mean_cen = cen.mean_halo_mass()
print(f"{M_mean_cen:.2e}")
print(f"{M_mean_sat:.2e}")

#%%

plt.plot(k_arr, pkt_gk, label=r'$P_{q_\perp,1} + P_{q_\perp,2}$', color='tab:red')
plt.plot(k_arr, pkt_gk1, label=r'$P_{q_\perp,1}$', color='tab:blue', linestyle='--')
plt.plot(k_arr, -pkt_gk2, label=r'$-P_{q_\perp,2}$', color='tab:cyan', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkp_gk, label=r'$P_{q_\parallel,1} + P_{q_\parallel,2}$', color='tab:red')
plt.plot(k_arr, pkp_gk1, label=r'$P_{q_\parallel,1}$', color='tab:blue', linestyle='--')
plt.plot(k_arr, pkp_gk2, label=r'$P_{q_\parallel,2}$', color='tab:cyan', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkt_gk, label=r'1-halo + 2-halo', color='tab:red')
plt.plot(k_arr, pkt_gk_1h, label=r'1-halo only', color='tab:purple', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_two_halo.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkt_gk, label=r'Central only', color='tab:red')
plt.plot(k_arr, pkt_gk_sat, label=r'Central + satellites', color='tab:purple', linestyle='--')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=12, frameon=False)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_satellites.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Calculate angular power spectra

#%%

ells = np.geomspace(2, 1e4, 256)

# Transverse auto- and cross-correlations
Clt_gk = kSZ.get_Cl(pkt_gk1, pkt_gk2, ells, kind="perp", ab="eg")
Clt_gg = kSZ.get_Cl(pkt_gg1, pkt_gg2, ells, kind="perp", ab="gg")
Clt_kk = kSZ.get_Cl(pkt_kk1, pkt_kk2, ells, kind="perp", ab="ee")

# Galaxy-kSZ cross-correlation (longitudinal)
Clp_gk = kSZ.get_Cl(pkp_gk1, pkp_gk2, ells, kind="par", ab="eg")

# Galaxy-kSZ cross-correlation (transverse) without two-halo term
Clt_gk_1h = kSZ.get_Cl(pkt_gk1_1h, pkt_gk2, ells, kind="perp", ab="eg")

# Galaxy-kSZ cross-correlation (transverse) with satellites
Clt_gk_sat = kSZ.get_Cl(pkt_gk1_sat, pkt_gk2, ells, kind="perp", ab="eg")

#%%

Clt_gk_T = Clt_gk[0] + Clt_gk[1]
Clt_gg_T = Clt_gg[0] + Clt_gg[1]
Clt_kk_T = Clt_kk[0] + Clt_kk[1]

Clp_gk_T = Clp_gk[0] + Clp_gk[1]

Clt_gk_T_1h = Clt_gk_1h[0] + Clt_gk_1h[1]
Clt_gk_T_sat = Clt_gk_sat[0] + Clt_gk_sat[1]

#%%

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_T), color="tab:blue", label=r'$D_{\ell, \perp, T}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$', linestyle='--')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:blue", label=r'$-D_{\ell, \perp, 2}$', linestyle='dotted')

plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk_T), color="tab:red", label=r'$D_{\ell, \parallel, T}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[0]), color="tab:red", label=r'$D_{\ell, \parallel, 1}$', linestyle='--')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[1]), color="tab:red", label=r'$D_{\ell, \parallel, 2}$', linestyle='dotted')

plt.xlim(40, 8e3)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=12, frameon=False, loc="center right", ncol=2)
#plt.savefig('kSZ_angular_power_spectra.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%
# Calculate covariance

#%%

# Galaxy noise
sigma_v = 300e3 / ccl.physical_constants.CLIGHT
ng_srad = 150 * (180/np.pi)**2 # galaxies per square radian
nl_gg = np.ones_like(ells) * sigma_v**2 / ng_srad

# CMB power spectrum
T_CMB_uK = cosmo['T_CMB'] * 1e6
d = np.loadtxt('data/camb_93159309_scalcls.dat', unpack=True)
ells_cmb = d[0]
D_ells_cmb = d[1]
C_ells_cmb = 2 * np.pi * D_ells_cmb / (ells_cmb * (ells_cmb+1) * T_CMB_uK**2)
nl_TT_cmb = interp1d(ells_cmb, C_ells_cmb, bounds_error=False, fill_value=0)(ells)

# Secondary anisotropies
d = pickle.load(open("data/P-ACT_theory_cells.pkl", "rb"))
ells_act = d["ell"]
D_ells_act = d["tt", "dr6_pa5_f090", "dr6_pa5_f090"] 
C_ells_act = 2 * np.pi * D_ells_act / (ells_act * (ells_act+1) * T_CMB_uK**2)
nl_TT_act = interp1d(ells_act, C_ells_act, bounds_error=False, fill_value=0)(ells)

# CMB noise
d = np.loadtxt("data/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_CMB.txt", unpack=True)
ells_n = d[0]
C_ells_n = d[1]
C_ells_n /= T_CMB_uK**2
nl_TT_cmb += interp1d(ells_n, C_ells_n, bounds_error=False, fill_value=(C_ells_n[0], C_ells_n[-1]))(ells)
nl_TT_act += interp1d(ells_n, C_ells_n, bounds_error=False, fill_value=(C_ells_n[0], C_ells_n[-1]))(ells)

#%%

f_sky = 0.5
var = ((Clt_gg_T + nl_gg) * (-Clt_kk_T + nl_TT_act) + Clt_gk_T**2) / ((2 * ells + 1) * f_sky)

#%%

plt.plot(ells, Clt_gk_T, color='tab:blue')
plt.errorbar(ells, Clt_gk_T, yerr=np.sqrt(var), color='tab:blue', alpha=0.5)
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$C_{\ell, \perp}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.show()

#%%
# Calculate signal-to-noise

#%%

def S_to_N(Cl1, Cl2, var):
    delta_ell = np.gradient(ells)
    Cl1 = np.abs(Cl1)
    Cl2 = np.abs(Cl2)
    s2n = np.sqrt(np.sum(delta_ell * (Cl1 - Cl2)**2 / var))
    return s2n
  
# kSZ-only
s2n_ksz = np.sqrt(np.sum(Clt_gk_T**2 / var))
print('S/N kSZ =', s2n_ksz)

# Sub-dominant contribution
s2n_sd = S_to_N(Clt_gk[0], Clt_gk[1], var)
print('S/N Perp_1 vs Perp_2 =', s2n_sd)

# Longitudinal mode
s2n_par = S_to_N(Clt_gk_T, Clp_gk_T, var)
print('S/N Perp vs Par =', s2n_par)

# Two-halo term
s2n_2h = S_to_N(Clt_gk[0], Clt_gk_1h[0], var)
print('S/N 1h+2h vs 1h =', s2n_2h)

# Satelite galaxies
s2n_sat = S_to_N(Clt_gk[0], Clt_gk_sat[0], var)
print('S/N cen+sat vs sat =', s2n_sat)

#%%
# Higher-order contributions

#%%

from scipy.interpolate import RectBivariateSpline

# Define grids for M and k
log10M_grid = np.linspace(11, 15, 128)
M_grid = 10**log10M_grid
k_grid = np.logspace(-3, 2, 128)
log10k_grid = np.log10(k_grid)

# Arrays for Fourier transforms
u_pM_grid = np.zeros((len(k_grid), len(M_grid)))
u_pG_grid = np.zeros((len(k_grid), len(M_grid)))
u_pE_grid = np.zeros((len(k_grid), len(M_grid)))

# Fill the grids
for i, k_val in enumerate(k_grid):
    for j, M_val in enumerate(M_grid):
        u_pM_grid[i, j] = pM.fourier(cosmo, k_val, M_val, a) / M_val
        u_pG_grid[i, j] = pG.fourier(cosmo, k_val, M_val, a) / pG.get_normalization(cosmo, a, hmc=hmc)
        u_pE_grid[i, j] = pGas.fourier(cosmo, k_val, M_val, a)

pM_interp = RectBivariateSpline(log10k_grid, log10M_grid, u_pM_grid)
pG_interp = RectBivariateSpline(log10k_grid, log10M_grid, u_pG_grid)
pE_interp = RectBivariateSpline(log10k_grid, log10M_grid, u_pE_grid)

k_dense = np.logspace(-3, 1, 1000)
P_dense = ccl.linear_matter_power(cosmo, k_dense, a)
P_L_interp = interp1d(k_dense, P_dense, bounds_error=False, fill_value=0.0)

#%%

test_k = 1e-2

ho = ksz.HigherOrder(cosmo, k_arr, a, M_vals, n_M, b_M, pM_interp, pG_interp, pE_interp, P_L_interp) 

Pt_tri_4h = ho.compute_P(test_k, spectra_type="trispectrum", kind="perp", term="4h") 

Pp_tri_1h = ho.compute_P(test_k, spectra_type="trispectrum", kind="par", term="1h") 
Pp_tri_4h = ho.compute_P(test_k, spectra_type="trispectrum", kind="par", term="4h") 

Pp_bi_1h_g = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="1h", density_kind='galaxy') 
Pp_bi_3h_g = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="3h", density_kind='galaxy')

Pp_bi_1h_e = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="1h", density_kind='electron') 
Pp_bi_3h_e = ho.compute_P(test_k, spectra_type="bispectrum", kind="par", term="3h", density_kind='electron')
