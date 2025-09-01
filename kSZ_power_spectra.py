import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.interpolate import interp1d

#%%
# Functions to compute the 3D power spectrum

#%%

def P_perp_1(k, a, pk_mm, pk_eg):
    
    mu_vals = np.linspace(-0.99, 0.99, 32)
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
    
    return integral / (2*np.pi)**2


def P_perp_2(k, a, pk_em, pk_gm):

    mu_vals = np.linspace(-0.99, 0.99, 32)
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
    
    return integral / (2*np.pi)**2
   

def P_par_1(k, a, pk_mm, pk_eg):

    mu_vals = np.linspace(-0.99, 0.99, 32)
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
    
    return integral / (2*np.pi)**2


def P_par_2(k, a, pk_em, pk_gm):

    mu_vals = np.linspace(-0.99, 0.99, 32)
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
    
    return integral / (2*np.pi)**2


def get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind, variant='full'):

        if kind == 'perp':
            pkf1 = P_perp_1
            pkf2 = P_perp_2
            
        elif kind == 'par':
            pkf1 = P_par_1
            pkf2 = P_par_2
            
        else:
            raise ValueError(f"Unknown power spectrum type {kind}")
            
        pk_mm = pk_dict['mm']['full']
        pk_em = pk_dict['em']['full']
        pk_gm = pk_dict['gm']['full']
        pk_eg = pk_dict['eg'][variant]

        pk1 = []
        pk2 = []
        for i, a in enumerate(a_arr):
            H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
            f = ccl.growth_rate(cosmo, a)
            aHf = a*H*f
            pk1.append(np.array([aHf**2 * pkf1(k, a, pk_mm, pk_eg) for k in k_arr]))
            pk2.append(np.array([aHf**2 * pkf2(k, a, pk_gm, pk_em) for k in k_arr]))
        
        pk1 = np.array(pk1)
        pk2 = np.array(pk2)
        
        lk_arr = np.log(k_arr)
        
        pkf1 = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk1, is_logp=False)
        pkf2 = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk2, is_logp=False)
        
        return pkf1, pkf2


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

k_arr = np.logspace(-3, 1, 24)
lk_arr = np.log(k_arr)
a_arr = np.linspace(0.1, 1, 32)

log10M = np.linspace(11, 15, 1000)
M = 10**log10M

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 16})

#%%
# Halo model implementation

#%%

hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pG = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=12.89, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1)
pG_cen = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=np.log10(1.03e13), log10M0_0=np.log10(1e14), log10M1_0=np.log10(1.2e14), alpha_0=1.1)
pG_sat = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=12.733, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1, fc_0=0.3)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
#pE = hp.HaloProfileDensityBattaglia(mass_def=hmd_200c)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

bg_cen = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=1e-4, a=1/(1+0.55), prof=pG_cen)
bg_sat = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=1e-4, a=1/(1+0.55), prof=pG_sat)

# Cross-correlations
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_eg = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pE, lk_arr=lk_arr, a_arr=a_arr)
pk_gg = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_2h=False)
pk_eg_2h = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG, lk_arr=lk_arr, a_arr=a_arr, get_1h=False, get_2h=True)
pk_eg_cen = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG_cen, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_sat = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG_sat, lk_arr=lk_arr, a_arr=a_arr)

#%%

z = 0.55
a = 1/(1+z)

# Plot the different contributions
plt.figure(figsize=(8, 6))
plt.plot(k_arr, pk_mm(k_arr, a), label=r'm-m', color="mediumblue")
plt.plot(k_arr, pk_gm(k_arr, a), label=r'g-m', color="crimson")
plt.plot(k_arr, pk_em(k_arr, a), label=r'e-m', color="deepskyblue")
plt.plot(k_arr, pk_eg(k_arr, a), label=r'e-g', color="blueviolet")
plt.plot(k_arr, pk_ee(k_arr, a), label=r'e-e', color="hotpink")
plt.plot(k_arr, pk_gg(k_arr, a), label=r'g-g', color="gold")
plt.loglog()
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k$', fontsize=28)
plt.ylabel(r'$P(k)$', fontsize=28)
plt.legend(fontsize=18, frameon=False, loc="lower left")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_power_spectra_test_plot.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# 3D power spectra calculations

#%%

pk_dict = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg, '1h': pk_eg_1h, '2h': pk_eg_2h, 'cen': pk_eg_cen, 'sat': pk_eg_sat},
           'gm': {'full': pk_gm},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg},
           'ee': {'full': pk_ee}}

# Perpendicular component
pkt1, pkt2 = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="perp")
pkt = pkt1 + pkt2

# Perpendicular component 1-halo only
pkt_1h, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="perp", variant="1h")

# Perpendicular component 2-halo only
pkt_2h, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="perp", variant="2h")

# Perpendicular component central-only
pkt_cen, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="perp", variant="cen")

# Perpendicular component with higher fraction of satellites
pkt_sat, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="perp", variant="sat")

# Parallel component
pkp1, pkp2 = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="par")
pkp = pkp1 + pkp2

# Parallel component 1-halo only
pkp_1h, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="par", variant="1h")

# Parallel component 2-halo only
pkp_2h, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="par", variant="2h")

#%%

z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = a*H*f

k_vals = np.array([
    1.00000000e-03, 1.49249555e-03, 2.22754295e-03, 3.32459793e-03,
    4.96194760e-03, 7.40568469e-03, 1.10529514e-02, 1.64964807e-02,
    2.46209240e-02, 3.67466194e-02, 5.48441658e-02, 8.18546731e-02,
    1.22167735e-01, 1.82334800e-01, 2.72133877e-01, 4.06158599e-01,
    6.06189899e-01, 9.04735724e-01, 1.35031404e+00, 2.01533769e+00,
    3.00788252e+00, 4.48925126e+00, 6.70018750e+00, 1.00000000e+01
])

pkt_tri_4h = np.array([
    5.13343086e-05, 5.13183965e-05, 5.50799044e-05, 6.84407783e-05,
    1.02876899e-04, 1.76714542e-04, 3.07700675e-04, 4.98411210e-04,
    7.51527524e-04, 1.08681836e-03, 1.35988974e-03, 1.49833148e-03,
    1.40328153e-03, 1.14242546e-03, 8.03099140e-04, 4.99815126e-04,
    2.70930639e-04, 1.30579702e-04, 5.61602156e-05, 2.22670384e-05,
    8.23329064e-06, 2.88355485e-06, 9.37244969e-07, 2.62201065e-07
])

pkp_bi_3h = np.array([
    6.56284242e-03, 9.41753711e-03, 1.32700753e-02, 1.82227718e-02,
    2.38080317e-02, 2.90466796e-02, 3.32674804e-02, 3.55644709e-02,
    3.44753142e-02, 2.89638601e-02, 1.94671794e-02, 1.15415948e-02,
    5.88856209e-03, 2.57768792e-03, 1.01103587e-03, 3.51048634e-04,
    1.07027854e-04, 2.93056627e-05, 7.21100127e-06, 1.60507822e-06,
    3.18935898e-07, 5.69620575e-08, 9.17793339e-09, 1.06736523e-09
])

pkp_bi_1h = np.array([
    3.99327985e-03, 3.99327786e-03, 3.99327343e-03, 3.99326357e-03,
    3.99324158e-03, 3.99319265e-03, 3.99308379e-03, 3.99284159e-03,
    3.99230257e-03, 3.99110248e-03, 3.98843267e-03, 3.98249635e-03,
    3.96930630e-03, 3.94005338e-03, 3.87566320e-03, 3.73656777e-03,
    3.44758486e-03, 2.89508764e-03, 2.00916314e-03, 1.00266898e-03,
    3.45366828e-04, 1.00237549e-04, 2.48069961e-05, 4.27496287e-06
])

pkp_tri_4h = np.array([
   -2.74589264e-05, -2.75712435e-05, -2.70575080e-05, -2.36572085e-05,
   -1.03999570e-05,  3.10765950e-05,  1.42379089e-04,  4.03333206e-04,
    9.27286919e-04,  1.73396984e-03,  2.46445163e-03,  3.01612940e-03,
    3.15806871e-03,  2.80160820e-03,  2.07898200e-03,  1.33893468e-03,
    7.15796824e-04,  3.29001075e-04,  1.31829705e-04,  4.79823993e-05,
    1.61359235e-05,  5.14540594e-06,  1.61659143e-06,  3.85996436e-07
])

pkp_tri_1h = np.array([
    3.21729788e-07, 7.16662163e-07, 1.59637442e-06, 3.55589804e-06,
    7.92051505e-06, 1.76415548e-05, 3.92894926e-05, 8.74827387e-05,
    1.94697737e-04, 4.32858979e-04, 9.60229802e-04, 2.12053112e-03,
    4.64080532e-03, 9.97848185e-03, 2.07485535e-02, 4.06556541e-02,
    7.25463308e-02, 1.13382592e-01, 1.46109229e-01, 1.38296637e-01,
    8.81480240e-02, 4.39681567e-02, 1.68431871e-02, 2.95159966e-03
])

k_arr_mm = np.logspace(-3, 1, 64)
lk_arr_mm = np.log(k_arr_mm)
pkp_mm = aHf**2 * pk_mm(k_arr_mm, a) / (k_arr_mm**2)

pkp_mm_interp = np.interp(k_arr, k_arr_mm, pkp_mm)
pkp_tot = pkp(k_arr, a) + pkp_bi_1h + pkp_bi_3h + pkp_mm_interp + pkp_tri_1h + pkp_tri_4h

#%%

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pkt1(k_arr, a), label=r'$P_{q_\perp,1}$', color='mediumblue', linewidth=2)
plt.plot(k_arr, pkt_1h(k_arr, a), label=r'$P_{q_\perp,1}^{\rm (1h)}$', color='mediumblue', linestyle='dashed')
plt.plot(k_arr, pkt_2h(k_arr, a), label=r'$P_{q_\perp,1}^{\rm (2h)}$', color='mediumblue', linestyle='dotted')
plt.plot(k_arr, -pkt2(k_arr, a), label=r'$-P_{q_\perp,2}$', color='deepskyblue', linewidth=2)
plt.plot(k_arr, pkt(k_arr, a), label=r'$P_{q_\perp, \rm tot}$', color='crimson', linewidth=2)
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=28)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=28)
plt.loglog()
plt.legend(fontsize=20, frameon=False, ncol=1, loc="lower left")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_power_spectrum_transverse.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pkp1(k_arr, a), label=r'$P_{q_\parallel,1}$', color='mediumblue', linewidth=2)
plt.plot(k_arr, pkp_1h(k_arr, a), label=r'$P_{q_\parallel,1}^{\rm (1h)}$', color='mediumblue', linestyle='dashed')
plt.plot(k_arr, pkp_2h(k_arr, a), label=r'$P_{q_\parallel,1}^{\rm (2h)}$', color='mediumblue', linestyle='dotted')
plt.plot(k_arr, pkp2(k_arr, a), label=r'$P_{q_\parallel,2}$', color='deepskyblue', linewidth=2)
plt.plot(k_arr, pkp(k_arr, a), label=r'$P_{q_\parallel,\rm tot}$', color='crimson', linewidth=2)
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=28)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=28)
plt.loglog()
plt.legend(fontsize=20, frameon=False, ncol=1, loc="lower left")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_power_spectrum_longitudinal.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pkt1(k_arr, a), label=r'$P_{q_\perp,1}$', color='mediumblue', linewidth=2)
plt.plot(k_arr, -pkt2(k_arr, a), label=r'$-P_{q_\perp,2}$', color='deepskyblue', linewidth=2)
plt.plot(k_arr, pkt_tri_4h, label=r'$P_{q_\perp, \rm c}$', color='blueviolet', linewidth=2)
plt.plot(k_arr, pkt(k_arr, a)+pkt_tri_4h, label=r'$P_{q_\perp, \rm tot}$', color='crimson', linewidth=2)
plt.xlim(1e-3, 1e1)
plt.ylim(5e-8, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=28)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=28)
plt.loglog()
plt.legend(fontsize=20, frameon=False, ncol=2, loc="upper center")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_power_spectrum_transverse_with_cng.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pkp1(k_arr, a), label=r'$P_{q_\parallel,1}$', color='mediumblue', linewidth=2)
plt.plot(k_arr, pkp2(k_arr, a), label=r'$-P_{q_\parallel,2}$', color='deepskyblue', linewidth=2)
plt.plot(k_arr, pkp_bi_1h, label=r'$P_{q_\parallel, B_{\rm 1h}}$', color='hotpink', linewidth=2)
plt.plot(k_arr, pkp_bi_3h, label=r'$P_{q_\parallel, B_{\rm 3h}}$', color='hotpink', linewidth=2, linestyle='--')
#plt.plot(k_arr, pkp_tri_1h, label=r'$P_{q_\parallel, T_{\rm 1h}}$', color='blueviolet', linewidth=2)
#plt.plot(k_arr, pkp_tri_4h, label=r'$P_{q_\parallel, T_{\rm 4h}}$', color='blueviolet', linewidth=2, linestyle='--')
plt.plot(k_arr_mm, pkp_mm, label=r'$P_{\rm mm}$', color='gold')
plt.plot(k_arr, pkp_tot, label=r'$P_{q_\parallel, \rm tot}$', color='crimson')
plt.xlim(1e-3, 1e1)
plt.ylim(0.9e-13, 9.9e6)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=28)
plt.ylabel(r'$P_{q_\parallel}^{\pi T}(k)$', fontsize=28)
plt.loglog()
plt.legend(fontsize=20, frameon=False, ncol=3, loc="upper center")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_power_spectrum_longitudinal_with_cng.pdf', format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pkt_cen(k_arr, a), label=r'$P_{q_\perp}^{\rm (cen)}$', color='crimson', linewidth=2)
plt.plot(k_arr, pkt_sat(k_arr, a), label=r'$P_{q_\perp}^{\rm (cen+sats)}$', color='mediumblue', linewidth=2)
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=28)
plt.ylabel(r'$P_{q_\perp}^{\pi T}(k)$', fontsize=28)
plt.loglog()
plt.legend(fontsize=20, frameon=False, ncol=1, loc="lower left")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_power_spectrum_transverse_satellites.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Galaxy and kSZ Tracers

#%%

ells = np.unique(np.geomspace(2, 10000, 256).astype(int)).astype(float)
perp_prefac = 0.5 * ells * (ells+1) / (ells+0.5)**2

xH = 0.76
sigmaT_over_mp = 8.30883107e-17
ne_times_mp = 0.5 * (1+xH) * cosmo['Omega_b'] * cosmo['h']**2 * ccl.physical_constants.RHO_CRITICAL
sigmaTne = ne_times_mp * sigmaT_over_mp

#zz = np.linspace(0, 2, 1024)
#nz = np.exp(-(0.5*(zz-0.55)/0.05)**2)

nz_data = np.load("data/dndzs_lrgs_zhou.npz")
zz = nz_data['z']
nz = nz_data['nz2']

kernel_g = ccl.get_density_kernel(cosmo, dndz=(zz,nz))

chis = ccl.comoving_radial_distance(cosmo, 1/(1+zz))

tkt = ccl.Tracer()
tgt = ccl.Tracer()

tkt.add_tracer(cosmo, kernel=(chis, sigmaTne*(1+zz)**2))
tgt.add_tracer(cosmo, kernel=kernel_g)

tkp = ccl.Tracer()
tgp = ccl.Tracer()

tkp.add_tracer(cosmo, kernel=(chis, sigmaTne*(1+zz)**2), der_bessel=1)
tgp.add_tracer(cosmo, kernel=kernel_g, der_bessel=1)

#%%
# Calculate the angular power spectra

#%%

def get_Dl(ells, Cl):
    return ells*(ells+1)*Cl/(2*np.pi)

clt1 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt1)
clt2 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt2)
clt = clt1 + clt2

clp1 = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pkp1)
clp2 = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pkp2)
clp = clp1 + clp2

clt_1h = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_1h)
clt_2h = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_2h)
clt_cen = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_cen)
clt_sat = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat)

clp_1h = perp_prefac * ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pkt_1h)
clp_2h = perp_prefac * ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pkt_2h)

#%%

H_arr = cosmo['h'] * ccl.h_over_h0(cosmo, a_arr) / ccl.physical_constants.CLIGHT_HMPC
f_arr = cosmo.growth_rate(a_arr)
aHf_arr = a_arr * H_arr * f_arr
aHf = np.array([aHf])

pk2d_bi_1h = (pkp_bi_1h / aHf[0]).reshape(24, 1) * aHf_arr.reshape(1, 32)
pk2d_bi_3h = (pkp_bi_3h / aHf[0]).reshape(24, 1) * aHf_arr.reshape(1, 32)
pk2d_ttri_4h = (pkt_tri_4h / aHf[0]).reshape(24, 1) * aHf_arr.reshape(1, 32)
pk2d_ptri_4h = (pkp_tri_4h / aHf[0]).reshape(24, 1) * aHf_arr.reshape(1, 32)
pk2d_ptri_1h = (pkp_tri_1h / aHf[0]).reshape(24, 1) * aHf_arr.reshape(1, 32)
pk2d_mm = ((aHf_arr**2).reshape(1,32) * pkp_mm.reshape(64, 1)) / ((2 * np.pi)**3 * (k_arr_mm**2).reshape(64, 1))

pk_1h = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk2d_bi_1h.T, is_logp=False)
pk_3h = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk2d_bi_3h.T, is_logp=False)
pk_t4h = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk2d_ttri_4h.T, is_logp=False)
pk_p4h = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk2d_ptri_4h.T, is_logp=False)
pk_p1h = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk2d_ptri_1h.T, is_logp=False)
Pk2D_mm = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr_mm, pk_arr=pk2d_mm.T, is_logp=False)

clp_bi_1h = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pk_1h)
clp_bi_3h = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pk_3h)
clt_tri_4h = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pk_t4h)
clp_tri_4h = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pk_p4h)
clp_tri_1h = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=pk_p1h)
clp_mm = ccl.angular_cl(cosmo, tgp, tkp, ells, p_of_k_a=Pk2D_mm)
clp_tot = clp + clp_mm + clp_bi_1h + clp_bi_3h + clp_tri_1h + clp_tri_4h

#%%

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt), color="mediumblue", label=r'$D_{\ell, \perp, T}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp), color="crimson", label=r'$D_{\ell, \parallel, T}$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt1), color="mediumblue", label=r'$D_{\ell, \perp, 1}$', linestyle='dashed')
plt.plot(ells, get_Dl(ells, -clp1), color="crimson", label=r'$D_{\ell, \parallel, 1}$', linestyle='dashed')
plt.plot(ells, get_Dl(ells, -clt2), color="mediumblue", label=r'$-D_{\ell, \perp, 2}$', linestyle='dotted')
plt.plot(ells, get_Dl(ells, -clp2), color="crimson", label=r'$D_{\ell, \parallel, 2}$', linestyle='dotted')
plt.xlim(2, 1e4)
plt.ylim(0.9e-19, 9e-8)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=28)
plt.ylabel(r'$D_{\ell}^{\pi T}$', fontsize=28)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, loc="upper center", ncol=3)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt1), color="mediumblue", label=r'$D_{\ell, \perp, 1}$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_1h), color="mediumblue", label=r'$D_{\ell, \perp, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, get_Dl(ells, clt_2h), color="mediumblue", label=r'$D_{\ell, \perp, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, get_Dl(ells, -clt2), color="deepskyblue", label=r'$-D_{\ell, \perp, 2}$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt), color="crimson", label=r'$D_{\ell, \perp, \rm tot}$', linewidth=2)
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=28)
plt.ylabel(r'$D_{\ell, \perp}^{\pi T}$', fontsize=28)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, loc="lower right", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_transverse.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, -clp1), color="mediumblue", label=r'$D_{\ell, \parallel, 1}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp_1h), color="mediumblue", label=r'$D_{\ell, \parallel, 1}^{\rm (1h)}$', linestyle='dashed')
plt.plot(ells, get_Dl(ells, -clp_2h), color="mediumblue", label=r'$D_{\ell, \parallel, 1}^{\rm (2h)}$', linestyle='dotted')
plt.plot(ells, get_Dl(ells, -clp2), color="deepskyblue", label=r'$D_{\ell, \parallel, 2}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp), color="crimson", label=r'$D_{\ell, \parallel, \rm tot}$', linewidth=2)
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=28)
plt.ylabel(r'$D_{\ell, \parallel}^{\pi T}$', fontsize=28)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, loc="best", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_longitudinal.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt1), color="mediumblue", label=r'$D_{\ell, \perp, 1}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clt2), color="deepskyblue", label=r'$-D_{\ell, \perp, 2}$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_tri_4h), color="blueviolet", label=r'$D_{\ell, \perp, \rm c}$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt+clt_tri_4h), color="crimson", label=r'$D_{\ell, \perp, \rm tot}$', linewidth=2)
plt.xlim(2, 1e4)
plt.ylim(5e-18, 1e-8)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=28)
plt.ylabel(r'$D_{\ell, \perp}^{\pi T}$', fontsize=28)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, loc="upper center", ncol=2)
#plt.savefig('kSZ_angular_power_spectra_transverse_with_cng.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, -clp1), color="mediumblue", label=r'$D_{\ell, \parallel, 1}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp2), color="deepskyblue", label=r'$D_{\ell, \parallel, 2}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp_bi_1h), color="hotpink", label=r'$D_{\ell, \parallel, B_{\rm 1h}}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp_bi_3h), color="hotpink", label=r'$D_{\ell, \parallel, B_{\rm 3h}}$', linewidth=2, linestyle='--')
#plt.plot(ells, get_Dl(ells, -clp_tri_1h), color="blueviolet", label=r'$D_{\ell, \parallel, T_{\rm 1h}}$', linewidth=2)
#plt.plot(ells, get_Dl(ells, -clp_tri_4h), color="blueviolet", label=r'$D_{\ell, \parallel, T_{\rm 4h}}$', linewidth=2, linestyle='--')
plt.plot(ells, get_Dl(ells, -clp_mm), color="gold", label=r'$D_{\ell, \rm mm}$', linewidth=2)
plt.plot(ells, get_Dl(ells, -clp_tot), color="crimson", label=r'$D_{\ell, \parallel, \rm tot}$', linewidth=2)
plt.xlim(2, 1e4)
plt.ylim(9e-32, 1e-8)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=28)
plt.ylabel(r'$D_{\ell, \parallel}^{\pi T}$', fontsize=28)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, loc="upper center", ncol=3)
#plt.savefig('kSZ_angular_power_spectra_longitudinal_with_cng.pdf',  format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt_cen), color="crimson", label=r'$D_{\ell, \perp}^{\rm (cen)}$', linewidth=2)
#plt.plot(ells, get_Dl(ells, clt1), color="deepskyblue", label=r'Central with low satellite fraction', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_sat), color="mediumblue", label=r'$D_{\ell, \perp}^{\rm (cen+sats)}$', linewidth=2)
plt.xlim(2, 1e4)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=28)
plt.ylabel(r'$D_{\ell, \perp}^{\pi T}$', fontsize=28)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=20, frameon=False, loc="lower right", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_transverse_satellites.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%
# Covariance calculation

#%%

# Auto-correlations
clt_gg = perp_prefac * ccl.angular_cl(cosmo, tgt, tgt, ells, p_of_k_a=pkt)
clt_kk = perp_prefac * ccl.angular_cl(cosmo, tkt, tkt, ells, p_of_k_a=pkt)

#%%

# Galaxy noise
sigma_v = 300e3 / ccl.physical_constants.CLIGHT
ng_srad = 150 * (180/np.pi)**2 # galaxies per square radian
nl_gg = np.ones_like(ells) * sigma_v**2 / ng_srad

# kSZ noise
T_CMB_uK = cosmo['T_CMB'] * 1e6
with_secondaries = True

if with_secondaries:
    act_cells = pkl.load(open("data/P-ACT_theory_cells.pkl", "rb"))
    l_cmb = act_cells['ell']
    Dl_cmb = act_cells[("tt", "dr6_pa5_f090", "dr6_pa5_f090")]

else:
    cmb_cells = np.loadtxt('data/camb_93159309_scalcls.dat', unpack=True)
    l_cmb = cmb_cells[0]
    Dl_cmb = cmb_cells[1]
    
Cl_cmb = 2 * np.pi * Dl_cmb / (l_cmb * (l_cmb+1) * T_CMB_uK**2)
nl_kk_irr = interp1d(l_cmb, Cl_cmb, bounds_error=False, fill_value=0)(ells)

# SO noise
d = np.loadtxt("data/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_CMB.txt", unpack=True)
ln = d[0]
nl_so = d[1]
nl_so /= T_CMB_uK**2
nl_kk_so = np.exp(interp1d(ln, np.log(nl_so), bounds_error=False, fill_value='extrapolate')(ells))

# S4 noise
lknee = 2154
aknee = -3.5 # Atmospheric noise
DT = 2.0 # Noise rms: 2 uK-arcmin
fwhm = 1.4 # Beam FWHM
beam = np.exp(-0.5*ln*(ln+1)*(np.radians(fwhm/60)/2.355)**2)
Nwhite = DT**2*(np.pi/180/60)**2 # White amplitude in uK^2 srad
nl_s4 = Nwhite * (1 + (ln/lknee)**aknee)/beam**2
nl_s4 /= T_CMB_uK**2
nl_s4 = 1/(1/nl_s4+1/nl_so)
nl_kk_s4 = np.exp(interp1d(ln, np.log(nl_s4), bounds_error=False, fill_value='extrapolate')(ells))

#%%

f_sky = 0.5

# CVL
var_irr = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr) + clt**2) / ((2*ells+1) * f_sky * np.gradient(ells))

# SO
var_so = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr+nl_kk_so) + clt**2) / ((2*ells+1) * f_sky * np.gradient(ells))

# S4
var_s4 = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr+nl_kk_s4) + clt**2) / ((2*ells+1) * f_sky * np.gradient(ells))

#%%

def S_to_N(Cl, var):
    s2n = np.sqrt(np.sum(Cl**2 / var))
    return s2n

var = var_so

# kSZ-only
sn_ksz = S_to_N(clt1+clt2, var)
print('S/N kSZ =', f"{sn_ksz:.4}")

# Sub-dominant contribution
sn_sd = S_to_N(clt2, var)
print('S/N <ev><gv> =', f"{sn_sd:.4}")

# Trispectrum contribution
#sn_tri = S_to_N(clt_tri_4h, var)
#print('S/N cng =', f"{sn_tri:.4}")

# Longitudinal mode
clp_tot = clp1 + clp2 #+ clp_bi_1h + clp_bi_3h + clp_mm
sn_par = S_to_N(clp_tot, var)
print('S/N parallel =', f"{sn_par:.4}")

# Twoe-halo term only
sn_1h = S_to_N(clt1-clt_1h, var)
print('S/N 2h =', f"{sn_1h:.4}")

# Satelite galaxies
sn_cen = S_to_N(clt_sat-clt_cen, var)
print('S/N cen+sat vs cen only =', f"{sn_cen:.4}")
