import numpy as np
import pyccl as ccl
import HaloProfiles as hp

#%%

cosmo = ccl.CosmologyVanillaLCDM()
z = 0.55
a = 1/(1+z)
H = cosmo['h'] * ccl.h_over_h0(cosmo, a) / ccl.physical_constants.CLIGHT_HMPC
f = cosmo.growth_rate(a)
aHf = a * H * f

#%%

hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM, fourier_analytic=True)
pG = ccl.halos.HaloProfileHOD(mass_def=hmd_200c, concentration=cM, log10Mmin_0=12.89, log10M0_0=12.92, log10M1_0=13.95, alpha_0=1.1, bg_0=2.04)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15., log10M_min=10., nM=32)

#%%

P_L_interp = cosmo.get_linear_power()

def P_L(k_vec, P_L_interp):
    k_norm = max(1E-4, np.sqrt(np.sum(k_vec**2)))
    return P_L_interp(k_norm, a, cosmo)

#%%

def F_2(k1_vec, k2_vec):
    k1_norm = np.linalg.norm(k1_vec)
    k2_norm = np.linalg.norm(k2_vec)
    dot_prod = np.dot(k1_vec, k2_vec)
    mu_12 = dot_prod / (k1_norm * k2_norm)
    return 5/7 + (mu_12/2) * ((k1_norm / k2_norm) + (k2_norm / k1_norm)) + (2/7) * mu_12**2

def G_2(k1_vec, k2_vec):
    k1_norm = np.linalg.norm(k1_vec)
    k2_norm = np.linalg.norm(k2_vec)
    dot_prod = np.dot(k1_vec, k2_vec)
    mu_12 = dot_prod / (k1_norm * k2_norm)
    return 3/7 + (mu_12/2) * ((k1_norm / k2_norm) + (k2_norm / k1_norm)) + (4/7) * mu_12**2

def Q(k1_vec, k2_vec, k3_vec):
    k123_vec = k1_vec + k2_vec + k3_vec
    k23_vec = k2_vec + k3_vec
    k1_norm = np.linalg.norm(k1_vec)
    k123_norm = np.linalg.norm(k123_vec)
    k23_norm = np.linalg.norm(k23_vec)
    prefactor_1 = (7 * np.dot(k123_vec, k1_vec)) / k1_norm**2
    term_1 = prefactor_1 * F_2(k2_vec, k3_vec)
    prefactor_2 = (7 * np.dot(k123_vec, k23_vec)) / k23_norm**2 + (2 * k123_norm**2 * np.dot(k23_vec, k1_vec)) / (k23_norm**2 * k1_norm**2)
    term_2 = prefactor_2 * G_2(k2_vec, k3_vec)
    return term_1 + term_2

def F_3(k1_vec, k2_vec, k3_vec):
    return (1/54) * (Q(k1_vec, k2_vec, k3_vec) + Q(k2_vec, k3_vec, k1_vec) + Q(k3_vec, k1_vec, k2_vec))
   
def B_tree_level(k, kp):
    k1 = k
    k2 = - kp
    k3 = - (k - kp)
    B_13 = 2 * F_2(k1, k2) * P_L(k1, P_L_interp) * P_L(k2, P_L_interp)
    B_23 = 2 * F_2(k2, k3) * P_L(k2, P_L_interp) * P_L(k3, P_L_interp)
    B_31 = 2 * F_2(k3, k1) * P_L(k3, P_L_interp) * P_L(k1, P_L_interp)
    return B_13 + B_23 + B_31

def bi_3h(k, kp):
    B_tree = B_tree_level(k, kp)
    k1 = k
    k2 = -kp
    k3 = - (k - kp)
    ks = [k1, k2, k3]    
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    I1 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[0], a=1/(1+0.55), prof=pM)
    I2 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[1], a=1/(1+0.55), prof=pM)
    I3 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[2], a=1/(1+0.55), prof=pM)
    I = I1 * I2 * I3
    return B_tree * I

#%%

n_k = 24
n_mu = 16
n_phi = 16

lk_min, lk_max = -3, 1
lkps = np.linspace(lk_min, lk_max, n_k)
lkpps = np.linspace(lk_min, lk_max, n_k)
kps = 10**lkps
kpps = 10**lkpps
mups = np.linspace(-0.99, 0.99, n_mu)
mupps = np.linspace(-0.99, 0.99, n_mu)
phis = np.linspace(0+1E-4, 2 * np.pi-1E-4, n_phi)
cos_phis = np.cos(phis)
sin_phis = np.sin(phis)

#%%

def P_bi(k_mag, aHf, bi_func, P_L_interp):
    k_vec = np.array([0, 0, k_mag])
    prefactor = aHf**2 / ((2 * np.pi)**3 * k_mag**2)
    
    integrand_kp = []
    for i, kp in enumerate(kps):
        integrand_mu = []
        for mu in mups:
            sin_theta = np.sqrt(1 - mu**2)
            kp_vecs = kp * np.array([sin_theta * cos_phis, sin_theta * sin_phis, mu * np.ones_like(phis)]).T
            B_vals = np.array([bi_func(k_vec, kp_vec) for kp_vec in kp_vecs])
                                    
            integral_phi = np.trapz(B_vals, x=phis)
            integrand_mu.append(mu * integral_phi)
        integral_mu = np.trapz(integrand_mu, x=mups)
        integrand_kp.append((kp/k_mag) * integral_mu * kp)
            
    result = np.trapz(integrand_kp, x=np.log(kps))
    return prefactor * result

#%%

kmin = 1E-3
kmax = 1E1
ks = np.geomspace(kmin, kmax, n_k)
logk = np.log(ks)

Pk_bi = []

for ik, k in enumerate(ks):
    print(ik)
    b = P_bi(k, aHf, bi_3h, P_L_interp)
    print(b)
    Pk_bi.append(b)
    
Pk_bi_3h = np.array(Pk_bi)

print('k_vals=', ks)
print("Pk_bi=", Pk_bi_3h)

#%%

from scipy.interpolate import RectBivariateSpline

log10M_vals = np.linspace(11, 15, 128)
M_vals = 10**log10M_vals

nM_vals = nM(cosmo, M_vals, a=1/(1+0.55))
bM_vals = bM(cosmo, M_vals, a=1/(1+0.55))

log10M_grid = np.linspace(11, 15, 128)
M_grid = 10**log10M_grid
k_grid = np.logspace(-3, 2, 128)
log10k_grid = np.log10(k_grid)

u_pM_grid = np.zeros((len(k_grid), len(M_grid)))
u_pG_grid = np.zeros((len(k_grid), len(M_grid)))
u_pE_grid = np.zeros((len(k_grid), len(M_grid)))

for i, k_val in enumerate(k_grid):
    for j, M_val in enumerate(M_grid):
        u_pM_grid[i, j] = pM.fourier(cosmo, k_val, M_val, a) / pM.get_normalization(cosmo, a, hmc=hmc)
        u_pG_grid[i, j] = pG.fourier(cosmo, k_val, M_val, a) / pG.get_normalization(cosmo, a, hmc=hmc)
        u_pE_grid[i, j] = pE.fourier(cosmo, k_val, M_val, a) / pE.get_normalization(cosmo, a, hmc=hmc)

interp_pM = RectBivariateSpline(log10k_grid, log10M_grid, u_pM_grid)
interp_pG = RectBivariateSpline(log10k_grid, log10M_grid, u_pG_grid)
interp_pE = RectBivariateSpline(log10k_grid, log10M_grid, u_pE_grid)

#%%

def bi_1h(k, kp):
    k1 = k
    k2 = -kp
    k3 = - (k - kp)
    ks = [k1, k2, k3]
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)

    u1_vals = interp_pM.ev(np.log10(k_norms[0]), log10M)
    u2_vals = interp_pM.ev(np.log10(k_norms[1]), log10M)
    u3_vals = interp_pG.ev(np.log10(k_norms[2]), log10M)

    integrand = nM_vals * bM_vals * u1_vals * u2_vals * u3_vals
    result = np.trapz(integrand, log10M)
    return result

#%%

kmin = 1E-3
kmax = 1E1
ks = np.geomspace(kmin, kmax, n_k)
logk = np.log(ks)

Pk_bi = []

for ik, k in enumerate(ks):
    print(ik)
    b = P_bi(k, aHf, bi_1h, P_L_interp)
    print(b)
    Pk_bi.append(b)
    
Pk_bi_1h = np.array(Pk_bi)

print('k_vals=', ks)
print("Pk_bi=", Pk_bi_1h)
