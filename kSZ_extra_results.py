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

k_arr = np.logspace(-3, 1, 128)
lk_arr = np.log(k_arr)
a_arr = np.linspace(0.1, 1, 32)

log10M = np.linspace(11, 15, 1000)
M = 10**log10M

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 14})

#%%
# Halo model implementation

#%%

hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pG = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=12.89, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

# Cross-correlations
pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm = ccl.halos.halomod_Pk2D(cosmo, hmc, pG, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_eg = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)

#%%
# Power spectra calculations

#%%

pk_dict = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg},
           'gm': {'full': pk_gm},
           'em': {'full': pk_em}}

pkt1, pkt2 = get_pk2d(cosmo, k_arr, a_arr, pk_dict, kind="perp")
pkt = pkt1 + pkt2

#%%

ells = np.unique(np.geomspace(2, 10000, 256).astype(int)).astype(float)
perp_prefac = 0.5 * ells * (ells+1) / (ells+0.5)**2

xH = 0.76
sigmaT_over_mp = 8.30883107e-17
ne_times_mp = 0.5 * (1+xH) * cosmo['Omega_b'] * cosmo['h']**2 * ccl.physical_constants.RHO_CRITICAL
sigmaTne = ne_times_mp * sigmaT_over_mp

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

clt1 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt1)
clt2 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt2)
clt = clt1 + clt2

#%%
# Vary baryonic effects

#%%

pE_eta = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.1)
pE_lMc = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=12.0, beta=0.6, A_star=0.03, eta_b=0.5)
pk_eg_eta = ccl.halos.halomod_Pk2D(cosmo, hmc, pE_eta, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)
pk_em_eta = ccl.halos.halomod_Pk2D(cosmo, hmc, pE_eta, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_lMc = ccl.halos.halomod_Pk2D(cosmo, hmc, pE_lMc, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)
pk_em_lMc = ccl.halos.halomod_Pk2D(cosmo, hmc, pE_lMc, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

pk_dict_eta = {'mm': {'full': pk_mm},
               'eg': {'full': pk_eg_eta},
               'gm': {'full': pk_gm},
               'em': {'full': pk_em_eta}}

pkt1_eta, pkt2_eta = get_pk2d(cosmo, k_arr, a_arr, pk_dict_eta, kind="perp")
pkt_eta = pkt1_eta + pkt2_eta

pk_dict_lMc = {'mm': {'full': pk_mm},
               'eg': {'full': pk_eg_lMc},
               'gm': {'full': pk_gm},
               'em': {'full': pk_em_lMc}}

pkt1_lMc, pkt2_lMc = get_pk2d(cosmo, k_arr, a_arr, pk_dict_lMc, kind="perp")
pkt_lMc = pkt1_lMc + pkt2_lMc

clt_eta = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt1_eta+pkt2_eta)
clt_lMc = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt1_lMc+pkt2_lMc)

def get_Dl(ells, Cl):
    return ells*(ells+1)*Cl/(2*np.pi)

#%%

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt), label=r'$\log_{10} M_{\rm c} = 14.0$, $\eta_{\rm b} = 0.5$', color='crimson', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_eta), label=r'$\log_{10} M_{\rm c} = 14.0$, $\eta_{\rm b} = 0.1$', color='mediumblue', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_lMc), label=r'$\log_{10} M_{\rm c} = 12.0$, $\eta_{\rm b} = 0.5$', color='deepskyblue', linewidth=2)
plt.xlim(2, 1e4)
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.loglog()
plt.legend(fontsize=18, frameon=False, ncol=1, loc="lower right")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_angular_power_spectrum_baryonic_effects.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Vary baryon model

#%%

pE_bat = hp.HaloProfileDensityBattaglia(mass_def=hmd_200c)
pk_eg_bat = ccl.halos.halomod_Pk2D(cosmo, hmc, pE_bat, prof2=pG, lk_arr=lk_arr, a_arr=a_arr)
pk_em_bat = ccl.halos.halomod_Pk2D(cosmo, hmc, pE_bat, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pk_eg(k_arr, 1/(1+0.55)), label=r'HE profile', color='crimson', linewidth=2)
plt.plot(k_arr, pk_eg_bat(k_arr, 1/(1+0.55)), label=r'Battaglia profile', color='mediumblue', linewidth=2)
plt.loglog()
plt.xlim(1e-3, 1e1)
plt.xlabel(r'$k \; [h \, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P_{\rm eg}(k)$', fontsize=24)
plt.legend(fontsize=18, frameon=False, ncol=1, loc="lower left")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.show()

#%%

pk_dict_bat = {'mm': {'full': pk_mm},
               'eg': {'full': pk_eg_bat},
               'gm': {'full': pk_gm},
               'em': {'full': pk_em_bat}}

pkt1_bat, pkt2_bat = get_pk2d(cosmo, k_arr, a_arr, pk_dict_bat, kind="perp")
pkt_bat = pkt1_bat + pkt2_bat

clt_bat = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_bat)

#%%

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt), label=r'HE profile', color='crimson', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_bat), label=r'Battaglia profile', color='mediumblue', linewidth=2)
plt.xlim(2, 1e4)
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.loglog()
plt.legend(fontsize=18, frameon=False, ncol=1, loc="lower right")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_angular_power_spectrum_baryon_model.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Velocity reconstruction effect 1 - galaxy bias

#%%

bg = 2.0
pk_mm_bg = pk_gm / bg
pk_em_bg = pk_eg / bg

pk_dict_bg = {'mm': {'full': pk_mm_bg},
              'eg': {'full': pk_eg},
              'gm': {'full': pk_gm},
              'em': {'full': pk_em_bg}}

pkt1_bg, pkt2_bg = get_pk2d(cosmo, k_arr, a_arr, pk_dict_bg, kind="perp")
pkt_bg = pkt1_bg + pkt2_bg

clt_bg = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt1_bg+pkt2_bg)

#%%

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt), label=r'without galaxy bias', color='deepskyblue', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_bg), label=r'with galaxy bias', color='mediumblue', linewidth=2, linestyle='dashed')
plt.xlim(2, 1e4)
plt.xlabel(r'$\ell$', fontsize=24)
plt.ylabel(r'$[\ell (\ell + 1) \, / \, 2 \pi] \, C_{\ell}^{\pi T}$', fontsize=24)
plt.loglog()
plt.legend(fontsize=16, frameon=False, ncol=1, loc="lower right")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_angular_power_spectrum_galaxy_bias.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Signal-to-noise calculation

#%%

# Auto-correlations
clt_gg = perp_prefac * ccl.angular_cl(cosmo, tgt, tgt, ells, p_of_k_a=pkt)
clt_kk = perp_prefac * ccl.angular_cl(cosmo, tkt, tkt, ells, p_of_k_a=pkt)

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

#%%

f_sky = 0.5
var = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr+nl_kk_so) + clt**2) / ((2*ells+1) * f_sky * np.gradient(ells))

def S_to_N(Cl, var):
    s2n = np.sqrt(np.sum(Cl**2 / var))
    return s2n

sn_ksz = S_to_N(clt_bg, var)
print('S/N kSZ =', f"{sn_ksz:.4}")

#%%
# Velocity reconstruction effect 2 - shot noise

#%%

f_rec = 1.0 # Vary to investigate the impact on S/N
nl_gg_rec = nl_gg * f_rec

var_rec = ((clt_gg+nl_gg_rec) * (clt_kk+nl_kk_irr+nl_kk_so) + (clt1+clt2)**2) / ((2*ells+1) * f_sky * np.gradient(ells))
  
sn_ksz = S_to_N(clt1+clt2, var_rec)
print('S/N kSZ =', f"{sn_ksz:.4}")

#%%

f_recs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 25.0])
SN_vals = np.array([34.23, 24.34, 19.91, 17.26, 15.44, 14.10, 13.06, 12.22, 11.52, 10.93, 8.93, 7.774, 6.92])

plt.figure(figsize=(8, 6))
plt.plot(f_recs, SN_vals, color='mediumblue', linewidth=3)
plt.xlabel(r'$f_{\rm rec}$', fontsize=24)
plt.ylabel(r'$S/N$', fontsize=24)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_SN_f_rec.pdf', format="pdf", bbox_inches="tight")
plt.show()

#%%
# Effect of angular multipole cut-off scale

#%%

ell_max = np.array([1e3, 1.25e3, 1.5e3, 2e3, 2.25e3, 2.5e3, 2.75e3, 3e3, 3.5e3, 4e3, 5e3, 6e3, 7e3, 8e3, 1e4])
SN_vals = np.array([2.56, 3.49, 4.58, 8.00, 10.2, 12.8, 15.5, 18.3, 23.1, 26.5, 30.4, 32.3, 33.3, 33.7, 34.2])

plt.figure(figsize=(8, 6))
plt.plot(ell_max, SN_vals, color='mediumblue', linewidth=3)
plt.xscale('log')
plt.xlabel(r'$\ell_{\rm max}$', fontsize=24)
plt.ylabel(r'$S/N$', fontsize=24)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
#plt.savefig('kSZ_SN_ell_max.pdf', format="pdf", bbox_inches="tight")
plt.show()
