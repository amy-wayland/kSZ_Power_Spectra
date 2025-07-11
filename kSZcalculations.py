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

# Electron-galaxy two-halo term only
pk_eg_2h = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_1h=False, get_2h=True)

# Electron-electron
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pGas, lk_arr=lk_arr, a_arr=a_arr)

# Galaxy-galaxy
pk_gg = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

# Electron-galaxy with satellites
log10M = np.linspace(11, 15, 1000)
M_vals = 10**log10M
n_M = nM(cosmo, M_vals, 1/(1+0.55))
b_M = bM(cosmo, M_vals, 1/(1+0.55))
sat = ksz.Satellites(M_vals, M0=1e11, M1=3e12, M_min=1.74e12, nM=n_M)
M_mean_sat = sat.mean_halo_mass()

pG_sat = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM, log10Mmin_0=np.log10(1.74e12), log10M0_0=np.log10(1e11), log10M1_0=np.log10(3e12), alpha_0=1.1)
pk_eg_sat = ccl.halos.halomod_Pk2D(cosmo, hmc, pGas, prof2=pG_sat, lk_arr=lk_arr, a_arr=a_arr)


pk_dict = {
    'mm': {'full': pk_mm},
    'eg': {'full': pk_eg, '1h': pk_eg_1h, '2h': pk_eg_2h, 'sat': pk_eg_sat},
    'gm': {'full': pk_gm},
    'em': {'full': pk_em},
    'gg': {'full': pk_gg},
    'ee': {'full': pk_ee},
    }

#%%
# Calculate 3D power spectra

#%%

# Initialise a kSZ object using kSZclass.py
kSZ = ksz.kSZclass(cosmo, k_arr, a_arr, pk_dict)

z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = np.array([a * H * f])
a_index = 0

# Contribution from the trispectrum term
pkt_tri_4h = np.array([np.float64(-7.203464751484216e-06), np.float64(-5.7067538883843824e-06), np.float64(-5.472405724350699e-06), np.float64(-5.213591532965721e-06), np.float64(-4.930317290848859e-06), np.float64(-4.621826040498523e-06), np.float64(-4.2864768892701526e-06), np.float64(-3.9237587965323565e-06), np.float64(-3.5294837371217343e-06), np.float64(-3.1085450954019245e-06), np.float64(-2.6565055696857724e-06), np.float64(-2.1735537466343777e-06), np.float64(-1.6580129530679487e-06), np.float64(-1.1104378597367306e-06), np.float64(-5.304723830944471e-07), np.float64(8.114294594088589e-08), np.float64(7.248948247555021e-07), np.float64(1.3994768879296172e-06), np.float64(2.1114729973716868e-06), np.float64(2.8409486762551362e-06), np.float64(3.5919305139210453e-06), np.float64(4.360910724431833e-06), np.float64(5.141872643766284e-06), np.float64(5.936746261148497e-06), np.float64(6.7233802907227564e-06), np.float64(7.503727950933803e-06), np.float64(8.2672064727978e-06), np.float64(9.0018190584575e-06), np.float64(9.687649272246492e-06), np.float64(1.0311650445379507e-05), np.float64(1.0869692350013253e-05), np.float64(1.1339269601798368e-05), np.float64(1.1732274192901684e-05), np.float64(1.2033896822262906e-05), np.float64(1.2237261559783755e-05), np.float64(1.2311972224950943e-05), np.float64(1.2267709309094856e-05), np.float64(1.2106368812800721e-05), np.float64(1.1848664863480247e-05), np.float64(1.1522474935430247e-05), np.float64(1.1150032266526389e-05), np.float64(1.0716046278531043e-05), np.float64(1.0189183956111537e-05), np.float64(9.618953924853044e-06), np.float64(9.011500238648682e-06), np.float64(8.442379355291473e-06), np.float64(7.935229392130558e-06), np.float64(7.501197858561677e-06), np.float64(7.051761738986145e-06), np.float64(6.529784269417755e-06), np.float64(5.956856926937488e-06), np.float64(5.3489427693849224e-06), np.float64(4.780106314561275e-06), np.float64(4.3003328443425506e-06), np.float64(3.898483904610495e-06), np.float64(3.4401563666821127e-06), np.float64(2.928115157591513e-06), np.float64(2.4415451266193986e-06), np.float64(2.03415952581533e-06), np.float64(1.7436676438662532e-06), np.float64(1.6032339750002785e-06), np.float64(1.5001074364640365e-06), np.float64(1.2844291311263217e-06), np.float64(1.025211522173256e-06), np.float64(7.995628696153652e-07), np.float64(6.279522048630457e-07), np.float64(5.173774714186038e-07), np.float64(4.843792381143517e-07), np.float64(4.367455533077555e-07), np.float64(3.510238473354614e-07), np.float64(2.795711092007742e-07), np.float64(2.174418799487736e-07), np.float64(1.6245283102816533e-07), np.float64(1.3256104661993902e-07), np.float64(1.3026786787845144e-07), np.float64(1.142681184966123e-07), np.float64(9.108867062371734e-08), np.float64(7.036111352544435e-08), np.float64(5.438451494278402e-08), np.float64(4.089008016584974e-08), np.float64(3.6324717227650926e-08), np.float64(3.5987129523338355e-08), np.float64(2.9631256784120384e-08), np.float64(2.3731131156044944e-08), np.float64(1.8804162920698525e-08), np.float64(1.4689372681941453e-08), np.float64(1.1702792104794811e-08), np.float64(1.0701376846646039e-08), np.float64(9.827232282933552e-09), np.float64(7.914374120738021e-09), np.float64(6.339805372081012e-09), np.float64(5.112750646895981e-09), np.float64(3.980647874713243e-09), np.float64(3.220786461438484e-09), np.float64(2.8689224259018308e-09), np.float64(2.4111353624962773e-09), np.float64(1.9042312046741277e-09), np.float64(1.5302445074591695e-09), np.float64(1.1997664187882932e-09), np.float64(9.098584436554879e-10), np.float64(7.09965846622967e-10), np.float64(6.078859095116526e-10), np.float64(4.989954135593855e-10), np.float64(4.0263087846965003e-10), np.float64(3.184019465863243e-10), np.float64(2.352578213863686e-10), np.float64(1.6711259209368221e-10), np.float64(1.2432094427632433e-10), np.float64(1.0321062522683034e-10), np.float64(8.520713542949185e-11), np.float64(6.676457417223497e-11), np.float64(4.9509043116267656e-11), np.float64(3.468802159657135e-11), np.float64(2.2860068028898634e-11), np.float64(1.6790072281080083e-11), np.float64(1.4290029725680141e-11), np.float64(1.1681871536928509e-11), np.float64(8.976920690218291e-12), np.float64(6.5989710691829395e-12), np.float64(4.24793824282761e-12), np.float64(2.6446039008488885e-12), np.float64(2.0593476598495574e-12), np.float64(1.8369468844626364e-12), np.float64(1.4852942647976853e-12), np.float64(1.1730206752582734e-12), np.float64(9.342043143091147e-13), np.float64(7.55134933057561e-13), np.float64(1.0972818008493456e-12)])

# Galaxy-kSZ cross-correlation (transverse)
pkt_gk1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg) for k in k_arr])
pkt_gk2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_gm, pk_em) for k in k_arr])
pkt_gk = pkt_gk1 + pkt_gk2 + pkt_tri_4h

# Galaxy-kSZ cross-correlation (longitudinal)
pkp_gk1 = np.array([kSZ.P_par_1(k, a, a_index, pk_mm, pk_eg) for k in k_arr])
pkp_gk2 = np.array([kSZ.P_par_2(k, a, a_index, pk_gm, pk_em) for k in k_arr])
pkp_gk = pkp_gk1 + pkp_gk2

# Galaxy-galaxy auto-correlation (transverse)
pkt_gg1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_gg) for k in k_arr])
pkt_gg2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_gm, pk_gm) for k in k_arr])
pkt_gg = pkt_gg1 + pkt_gg2

# kSZ-kSZ auto-correlation (transverse)
pkt_kk1 = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_ee) for k in k_arr])
pkt_kk2 = np.array([kSZ.P_perp_2(k, a, a_index, pk_em, pk_em) for k in k_arr])
pkt_kk = pkt_kk1 + pkt_kk2

# Galaxy-kSZ cross-correlation (transverse) for one-halo term only
pkt_gk1_1h = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg_1h) for k in k_arr])

# Galaxy-kSZ cross-correlation (transverse) for one-halo term only
pkt_gk1_2h = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg_2h) for k in k_arr])

# Galaxy-kSZ cross-correlation (longitudinal) for one-halo term only
pkp_gk1_1h = np.array([kSZ.P_par_1(k, a, a_index, pk_mm, pk_eg_1h) for k in k_arr])

# Galaxy-kSZ cross-correlation (longitudinal) for one-halo term only
pkp_gk1_2h = np.array([kSZ.P_par_1(k, a, a_index, pk_mm, pk_eg_2h) for k in k_arr])

# Galaxy-kSZ cross-correlation (transverse) with satellites
pkt_gk1_sat = np.array([kSZ.P_perp_1(k, a, a_index, pk_mm, pk_eg_sat) for k in k_arr])

# Compare M_mean to central only case using the HOD parameters from pG
cen = ksz.Satellites(M_vals, M0=10**(12.92), M1=10**(13.95), M_min=10**(12.89), nM=n_M)
M_mean_cen = cen.mean_halo_mass()
print(f"{M_mean_cen:.2e}")
print(f"{M_mean_sat:.2e}")

#%%

plt.plot(k_arr, pkt_gk1, label=r'$P_{q_\perp,1}$', color='tab:blue')
plt.plot(k_arr, pkt_gk1_1h, label=r'$P_{q_\perp,1}^{\rm (1h)}$', color='tab:blue', linestyle='dashed')
plt.plot(k_arr, pkt_gk1_2h, label=r'$P_{q_\perp,1}^{\rm (2h)}$', color='tab:blue', linestyle='dotted')
plt.plot(k_arr, -pkt_gk2, label=r'$-P_{q_\perp,2}$', color='tab:cyan')
plt.plot(k_arr,pkt_tri_4h, label=r'$P_{q_\perp, \rm c}$', color='tab:purple')
plt.plot(k_arr, pkt_gk, label=r'$P_{q_\perp, \rm tot}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=14, frameon=False, ncol=2, loc="best")
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_transverse.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.plot(k_arr, pkp_gk1, label=r'$P_{q_\parallel,1}$', color='tab:blue')
plt.plot(k_arr, pkp_gk1_1h, label=r'$P_{q_\parallel,1}^{\rm (1h)}$', color='tab:blue', linestyle='dashed')
plt.plot(k_arr, pkp_gk1_2h, label=r'$P_{q_\parallel,1}^{\rm (2h)}$', color='tab:blue', linestyle='dotted')
plt.plot(k_arr, pkp_gk2, label=r'$P_{q_\parallel,2}$', color='tab:cyan')
plt.plot(k_arr, pkp_gk, label=r'$P_{q_\parallel,\rm tot}$', color='tab:red')
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=20)
plt.loglog()
plt.legend(fontsize=14, frameon=False, ncol=1)
plt.tick_params(which='both', direction='in', width=1, length=3)
#plt.savefig('kSZ_power_spectrum_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Calculate angular power spectra

#%%

ells = np.geomspace(2, 1e4, 256)

# Galaxy-electron cross-correlation (transverse)
Clt_gk = kSZ.get_Cl(ells, kind="perp", ab="eg")

# Galaxy-kSZ cross-correlation (transverse) for one-halo term only
Clt_gk_1h = kSZ.get_Cl(ells, kind="perp", ab="eg", variant="1h")

# Galaxy-kSZ cross-correlation (transverse) for two-halo term only
Clt_gk_2h = kSZ.get_Cl(ells, kind="perp", ab="eg", variant="2h")

# Galaxy-kSZ cross-correlation (transverse) with satellites
Clt_gk_sat = kSZ.get_Cl(ells, kind="perp", ab="eg", variant="sat")

#%%

# Galaxy-kSZ cross-correlation (longitudinal)
Clp_gk = kSZ.get_Cl(ells, kind="par", ab="eg")

# Galaxy-kSZ cross-correlation (longitudinal) for one-halo term only
Clp_gk_1h = kSZ.get_Cl(ells, kind="par", ab="eg", variant="1h")

# Galaxy-kSZ cross-correlation (longitudinal) for two-halo term only
Clp_gk_2h = kSZ.get_Cl(ells, kind="par", ab="eg", variant="2h")

#%%

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]+Clt_gk[1]), color="tab:blue", label=r'$D_{\ell, \perp, T}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$', linestyle='--')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:blue", label=r'$-D_{\ell, \perp, 2}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -(Clp_gk[0]+Clp_gk[1])), color="tab:red", label=r'$D_{\ell, \parallel, T}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[0]), color="tab:red", label=r'$D_{\ell, \parallel, 1}$', linestyle='--')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[1]), color="tab:red", label=r'$D_{\ell, \parallel, 2}$', linestyle='dotted')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=14, frameon=False, loc="center right", ncol=2)
plt.show()

#%%

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_1h[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_2h[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:cyan", label=r'$-D_{\ell, \perp, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]+Clt_gk[1]), color="tab:red", label=r'$D_{\ell, \perp, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=14, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_transverse.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk_1h[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk_2h[0]), color="tab:blue", label=r'$D_{\ell, \parallel, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -Clp_gk[1]), color="tab:cyan", label=r'$D_{\ell, \parallel, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, -(Clp_gk[0]+Clp_gk[1])), color="tab:red", label=r'$D_{\ell, \parallel, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=14, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_longitudinal.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%

pk2d_tri_4h = (pkt_tri_4h / aHf[0]).reshape(128, 1) * aHf_arr.reshape(1, 32)

tg, tk = kSZ._get_tracers(kind='perp')
prefac = ells * (ells+1) / (ells+0.5)**2

pk_tri_arr = pk2d_tri_4h.T
pk_tri = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_tri_arr, is_logp=False)
Clt_tri_4h = prefac * ccl.angular_cl(cosmo, tg, tk, ells, p_of_k_a=pk_tri)

plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_1h[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk_2h[0]), color="tab:blue", label=r'$D_{\ell, \perp, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, kSZ.get_Dl(ells, -Clt_gk[1]), color="tab:cyan", label=r'$-D_{\ell, \perp, 2}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_tri_4h), color="tab:purple", label=r'$D_{\ell, \perp, \rm c}$')
plt.plot(ells, kSZ.get_Dl(ells, Clt_gk[0]+Clt_gk[1]), color="tab:red", label=r'$D_{\ell, \perp, \rm tot}$')
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=20)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=20)
plt.tick_params(which='both', direction='in', width=1, length=3)
plt.legend(fontsize=14, frameon=False, loc="best", ncol=2)
#plt.savefig('kSZ_angular_power_spectra_transverse_with_cng.pdf',  format="pdf", bbox_inches="tight")
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

# Transverse auto-correlations
Clt_gg = kSZ.get_Cl(ells, kind="perp", ab="gg")
Clt_kk = kSZ.get_Cl(ells, kind="perp", ab="ee")

Clt_gk_T = Clt_gk[0] + Clt_gk[1]
Clt_gg_T = Clt_gg[0] + Clt_gg[1]
Clt_kk_T = Clt_kk[0] + Clt_kk[1]

f_sky = 0.5

# Knox formula to calculate variance
var = ((Clt_gg_T + nl_gg) * (-Clt_kk_T + nl_TT_act) + (Clt_gk[0]+Clt_gk[1])**2) / ((2 * ells + 1) * f_sky)

plt.plot(ells, (Clt_gk[0]+Clt_gk[1]), color='tab:blue')
plt.errorbar(ells, (Clt_gk[0]+Clt_gk[1]), yerr=np.sqrt(var), color='tab:blue', alpha=0.5)
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
    s2n = np.sqrt(np.sum(delta_ell * (Cl1 - Cl2)**2 / var))
    return s2n
  
# kSZ-only
sn_ksz = np.sqrt(np.sum(Clt_gk_T**2 / var))
print('S/N kSZ =', sn_ksz)

# Sub-dominant contribution
sn_sd = S_to_N(Clt_gk[0], Clt_gk[1], var)
print('S/N Perp_1 vs Perp_2 =', sn_sd)

# Longitudinal mode
sn_par = S_to_N((Clt_gk[0]+Clt_gk[1]), (Clp_gk[0]+Clp_gk[1]), var)
print('S/N Perp vs Par =', sn_par)

# One-halo term only
sn_1h = S_to_N(Clt_gk[0], Clt_gk_1h[0], var)
print('S/N 1h+2h vs 1h =', sn_1h)

# Trispectrum contribution
sn_tri = S_to_N(Clt_gk[0], Clt_tri_4h, var)
print('S/N Total vs Trispectrum Term =', sn_tri)

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
