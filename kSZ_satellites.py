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
    "font.size": 20})

#%%
# Halo model implementation

#%%

hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

pk_mm = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=lk_arr, a_arr=a_arr)
pk_em = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_ee = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pE, lk_arr=lk_arr, a_arr=a_arr)

#%%
# Satellites calculations

#%%

a = 1/(1+0.55)
M_vals = np.logspace(11, 15, 200)
dndlog10M = nM(cosmo, M_vals, a)
dndM = dndlog10M / (M_vals * np.log(10))


def get_mean_ncen(log10M, prof):
    return prof._Nc(10**log10M, a)*prof._fc(a)

def get_mean_nsat(log10M, prof):
    Nc = prof._Nc(10**log10M, a)
    Ns = prof._Ns(10**log10M, a)
    return Nc*Ns

def compute_satellite_fraction(cosmo, nM, prof):
    log10M = np.linspace(11, 15, 500)
    M = 10**log10M

    dndlog10M = nM(cosmo, M, a)
    
    N_cen = get_mean_ncen(log10M, prof)
    N_sat = get_mean_nsat(log10M, prof)
    
    dlogM = log10M[1] - log10M[0]
    
    N_cen_total = np.sum(dndlog10M * N_cen * dlogM)
    N_sat_total = np.sum(dndlog10M * N_sat * dlogM)

    f_sat = N_sat_total / (N_cen_total + N_sat_total)
    return f_sat

#%%

lMmin1 = 12.89
pG1 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin1, log10M0_0=lMmin1, log10M1_0=13.95, alpha_0=1.1, fc_0=1.0)
bg1 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG1)
fs1 = compute_satellite_fraction(cosmo, nM, pG1)
print(bg1, fs1)

lMmin2 = 12.87
pG2 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin2, log10M0_0=lMmin2, log10M1_0=13.95, alpha_0=1.1, fc_0=0.8)
bg2 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG2)
fs2 = compute_satellite_fraction(cosmo, nM, pG2)
print(bg2, fs2)

lMmin3 = 12.83
pG3 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin3, log10M0_0=lMmin3, log10M1_0=13.95, alpha_0=1.1, fc_0=0.6)
bg3 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG3)
fs3 = compute_satellite_fraction(cosmo, nM, pG3)
print(bg3, fs3)

lMmin4 = 12.77
pG4 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin4, log10M0_0=lMmin4, log10M1_0=13.95, alpha_0=1.1, fc_0=0.4)
bg4 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG4)
fs4 = compute_satellite_fraction(cosmo, nM, pG4)
print(bg4, fs4)

lMmin5 = 12.63
pG5 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin5, log10M0_0=lMmin5, log10M1_0=13.95, alpha_0=1.1, fc_0=0.2)
bg5 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG5)
fs5 = compute_satellite_fraction(cosmo, nM, pG5)
print(bg5, fs5)

lMmin6 = 12.45
pG6 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin6, log10M0_0=lMmin6, log10M1_0=13.95, alpha_0=1.1, fc_0=0.1)
bg6 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG6)
fs6 = compute_satellite_fraction(cosmo, nM, pG6)
print(bg6, fs6)

lMmin7 = 11.15
pG7 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin7, log10M0_0=lMmin7, log10M1_0=13.95, alpha_0=1.1, fc_0=0.0)
bg7 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG7)
fs7 = compute_satellite_fraction(cosmo, nM, pG7)
print(bg7, fs7)

lMmin8 = 12.85
pG8 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin8, log10M0_0=lMmin8, log10M1_0=13.95, alpha_0=1.1, fc_0=0.7)
bg8 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG8)
fs8 = compute_satellite_fraction(cosmo, nM, pG8)
print(bg8, fs8)

lMmin9 = 12.42
pG9 = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=lMmin9, log10M0_0=lMmin9, log10M1_0=13.95, alpha_0=1.1, fc_0=0.08)
bg9 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, 1E-4, a, pG9)
fs9 = compute_satellite_fraction(cosmo, nM, pG9)
print(bg9, fs9)

#%%

pk_eg_1 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG1, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_2 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG2, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_3 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG3, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_4 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG4, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_5 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG5, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_6 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG6, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_7 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG7, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_8 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG8, lk_arr=lk_arr, a_arr=a_arr)
pk_eg_9 = ccl.halos.halomod_Pk2D(cosmo, hmc, pE, prof2=pG9, lk_arr=lk_arr, a_arr=a_arr)

pk_gm_1 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG1, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_2 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG2, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_3 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG3, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_4 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG4, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_5 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG5, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_6 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG6, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_7 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG7, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_8 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG8, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)
pk_gm_9 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG9, prof2=pM, lk_arr=lk_arr, a_arr=a_arr)

pk_gg_1 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG1, prof2=pG1, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_2 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG2, prof2=pG2, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_3 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG3, prof2=pG3, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_4 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG4, prof2=pG4, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_5 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG5, prof2=pG5, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_6 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG6, prof2=pG6, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_7 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG7, prof2=pG7, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_8 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG7, prof2=pG8, lk_arr=lk_arr, a_arr=a_arr)
pk_gg_9 = ccl.halos.halomod_Pk2D(cosmo, hmc, pG7, prof2=pG9, lk_arr=lk_arr, a_arr=a_arr)

#%%

pk_dict1 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_1},
           'gm': {'full': pk_gm_1},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_1},
           'ee': {'full': pk_ee}}

pkt_sat1, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict1, kind="perp")

pk_dict2 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_2},
           'gm': {'full': pk_gm_2},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_2},
           'ee': {'full': pk_ee}}

pkt_sat2, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict2, kind="perp")

pk_dict3 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_3},
           'gm': {'full': pk_gm_3},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_3},
           'ee': {'full': pk_ee}}

pkt_sat3, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict3, kind="perp")

pk_dict4 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_4},
           'gm': {'full': pk_gm_4},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_4},
           'ee': {'full': pk_ee}}

pkt_sat4, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict4, kind="perp")

pk_dict5 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_5},
           'gm': {'full': pk_gm_5},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_5},
           'ee': {'full': pk_ee}}

pkt_sat5, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict5, kind="perp")

pk_dict6 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_6},
           'gm': {'full': pk_gm_6},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_6},
           'ee': {'full': pk_ee}}

pkt_sat6, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict6, kind="perp")

pk_dict7 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_7},
           'gm': {'full': pk_gm_7},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_7},
           'ee': {'full': pk_ee}}

pkt_sat7, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict7, kind="perp")

pk_dict8 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_8},
           'gm': {'full': pk_gm_8},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_8},
           'ee': {'full': pk_ee}}

pkt_sat8, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict8, kind="perp")

pk_dict9 = {'mm': {'full': pk_mm},
           'eg': {'full': pk_eg_9},
           'gm': {'full': pk_gm_9},
           'em': {'full': pk_em},
           'gg': {'full': pk_gg_9},
           'ee': {'full': pk_ee}}

pkt_sat9, _ = get_pk2d(cosmo, k_arr, a_arr, pk_dict9, kind="perp")

#%%

plt.figure(figsize=(8, 6))
plt.plot(k_arr, pkt_sat2(k_arr, a)/pkt_sat1(k_arr, a), label=r'$f_{\rm sat} = 0.16$', color='hotpink', linewidth=2)
plt.plot(k_arr, pkt_sat3(k_arr, a)/pkt_sat1(k_arr, a), label=r'$f_{\rm sat} = 0.19$', color='blueviolet', linewidth=2)
plt.plot(k_arr, pkt_sat4(k_arr, a)/pkt_sat1(k_arr, a), label=r'$f_{\rm sat} = 0.24$', color='mediumblue', linewidth=2)
plt.plot(k_arr, pkt_sat5(k_arr, a)/pkt_sat1(k_arr, a), label=r'$f_{\rm sat} = 0.34$', color='deepskyblue', linewidth=2)
plt.plot(k_arr, pkt_sat6(k_arr, a)/pkt_sat1(k_arr, a), label=r'$f_{\rm sat} = 0.43$', color='gold', linewidth=2)
plt.plot(k_arr, pkt_sat7(k_arr, a)/pkt_sat1(k_arr, a), label=r'$f_{\rm sat} = 1.0$', color='crimson', linewidth=2)
plt.xticks([1, 10])
plt.xlim(9e-1, 1e1)
plt.ylim(9.5e-2, 1.2e0)
plt.xlabel(r'$k \; [\mathrm{Mpc}^{-1}]$', fontsize=32)
plt.ylabel(r'$P(k, f_{\rm sat}) \,/\, P(k, f_{\rm sat} = 0.13)$', fontsize=32)
plt.loglog()
plt.legend(fontsize=22, frameon=False, ncol=1, loc="lower left")
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
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

clt_sat1 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat1)
clt_sat2 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat2)
clt_sat3 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat3)
clt_sat4 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat4)
clt_sat5 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat5)
clt_sat6 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat6)
clt_sat7 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat7)
clt_sat8 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat8)
clt_sat9 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt_sat9)

#%%

plt.figure(figsize=(8, 6))
plt.plot(ells, get_Dl(ells, clt_sat3)/get_Dl(ells, clt_sat1), color="crimson", label=r'$f_{\rm sat} = 0.2$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_sat5)/get_Dl(ells, clt_sat1), color="hotpink", label=r'$f_{\rm sat} = 0.3$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_sat6)/get_Dl(ells, clt_sat1), color="blueviolet", label=r'$f_{\rm sat} = 0.4$', linewidth=2)
plt.plot(ells, get_Dl(ells, clt_sat7)/get_Dl(ells, clt_sat1), color="mediumblue", label=r'$f_{\rm sat} = 1.0$', linewidth=2)
plt.xlim(0.9e3, 1e4)
plt.ylim(5.8e-1, 1.1e0)
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=32)
plt.ylabel(r'$D_{\ell}(f_{\rm sat}) \;/\; D_{\ell}(f_{\rm sat}=0.14)$', fontsize=32)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=24, frameon=False, loc="lower left", ncol=1)
#plt.savefig('kSZ_angular_power_spectra_satellites.pdf',  format="pdf", bbox_inches="tight")
plt.show()

#%%
# Covariance calculation

#%%

pkt1, pkt2 = get_pk2d(cosmo, k_arr, a_arr, pk_dict1, kind="perp")
pkt = pkt1 + pkt2

clt1 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt1)
clt2 = perp_prefac * ccl.angular_cl(cosmo, tgt, tkt, ells, p_of_k_a=pkt2)
clt_gk = clt1 + clt2

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

# Beam
fwhm = 1.4 # arcmins
fwhm = 1.4 * np.pi / (60 * 180) # radians
sigma = fwhm / 2.355
bl = np.exp(-0.5 * ells * (ells + 1) * sigma**2) # multiply Cl by bl

#%%

f_sky = 0.5

# CVL
var_irr = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr) + clt_gk**2) / ((2*ells+1) * f_sky * np.gradient(ells))

# SO
var_so = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr+nl_kk_so) + clt_gk**2) / ((2*ells+1) * f_sky * np.gradient(ells))

# S4
var_s4 = ((clt_gg+nl_gg) * (clt_kk+nl_kk_irr+nl_kk_s4) + clt_gk**2) / ((2*ells+1) * f_sky * np.gradient(ells))

#%%

def S_to_N(Cl, var):
    s2n = np.sqrt(np.sum(Cl**2 / var))
    return s2n

var = var_so

sn_sat = S_to_N(clt_sat1-clt_sat7, var)
print('S/N satellites =', f"{sn_sat:.4}")

#%%

sn_sat1 = S_to_N(clt_sat1, var)
sn_sat7 = S_to_N(clt_sat7, var)
sn_sat = sn_sat1-sn_sat7
print('S/N satellites =', f"{sn_sat:.4}")

#%%

f_sat_vals = np.array([0.14, 0.16, 0.17, 0.19, 0.25, 0.34, 0.43, 0.48, 1.0])
sn_so = np.array([0.0, 0.27, 0.57, 0.85, 1.57, 2.94, 4.49, 4.8, 5.10])
sn_s4 = np.array([0.0, 0.74, 1.5, 2.2, 4.0, 7.1, 10.0, 10.6, 12.1])
sn_cvl = np.array([0.0, 2.4, 4.8, 7.0, 12.8, 22.5, 30.5, 31.9, 37.7])

plt.figure(figsize=(8, 6))
plt.plot(f_sat_vals, sn_so, color='crimson', linewidth=2, label=r'$\text{SO}$')
plt.plot(f_sat_vals, sn_s4, color='mediumblue', linewidth=2, label=r'$\text{S4}$')
plt.plot(f_sat_vals, sn_cvl, color='deepskyblue', linewidth=2, label=r'$\text{CVL}$')
plt.yscale('log')
plt.xlim(0.14, 1.0)
#plt.ylim(0.0, 1e1)
plt.xlabel(r'$f_{\rm sat}$', fontsize=30)
plt.ylabel(r'$S/N  \,(C_{\ell}^{\rm ge}(f_{\rm sat} = 0.14) - C_{\ell}^{\rm ge}(f_{\rm sat}))$', fontsize=26)
plt.tick_params(which='both', top=True, right=True, direction='in', width=1, length=5)
plt.legend(fontsize=24, frameon=False, ncol=1, loc="lower right")
plt.savefig('kSZ_SN_vs_f_sat.pdf', format="pdf", bbox_inches="tight")
plt.show()
