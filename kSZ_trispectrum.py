import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import scipy.integrate as si

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

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
    
def T_1122(k, kp, kpp):
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]
    
    total = 0.0
    indices = [
        (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2),
        (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 3, 0), (2, 3, 1)]

    for i, j, l in indices:
        q_vec = ks[i] + ks[l]
        F2_1 = F_2(q_vec, -ks[i])
        F2_2 = F_2(q_vec, ks[j])
        P1 = P_L(q_vec, P_L_interp)
        P2 = P_L(ks[i], P_L_interp)
        P3 = P_L(ks[j], P_L_interp)
        total += F2_1 * F2_2 * P1 * P2 * P3
    return 4 * total

def T_1113(k, kp, kpp):
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]

    total = 0.0
    indices = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]

    for i, j, l in indices:
        F3_val = F_3(ks[i], ks[j], ks[l])
        P1 = P_L(ks[i], P_L_interp)
        P2 = P_L(ks[j], P_L_interp)
        P3 = P_L(ks[l], P_L_interp)
        total += F3_val * P1 * P2 * P3
    return 6 * total

def T_tree_level(k, kp, kpp):
    return T_1122(k, kp, kpp) + T_1113(k, kp, kpp)

def tri_4h(k, kp, kpp):
    T_tree = T_tree_level(k, kp, kpp)
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    I1 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[0], a=1/(1+0.55), prof=pG)
    I2 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[1], a=1/(1+0.55), prof=pE)
    I3 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[2], a=1/(1+0.55), prof=pM)
    I4 = ccl.halos.pk_1pt.halomod_bias_1pt(cosmo, hmc, k=k_norms[3], a=1/(1+0.55), prof=pM)
    I = I1 * I2 * I3 * I4
    #I = 1.0
    return T_tree * I

#%%

n_k = 36
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

def P_T_perp(k_mag, aHf, tri_func, P_L_interp):
    k_vec = np.array([0, 0, k_mag])
    prefactor = aHf**2 / (2*np.pi)**5

    integrand_kp = []
    for i, kp in enumerate(kps):
        #print(i, n_k)
        integrand_kpp = []
        for j, kpp in enumerate(kpps):
            #print("  ", j)
            integrand_mu1 = []
            for mu1 in mups:
                sin_theta1 = np.sqrt(1 - mu1**2)
                kp_vec = kp * np.array([sin_theta1, 0.0, mu1])
                
                integrand_mu2 = []
                for mu2 in mupps:
                    sin_theta2 = np.sqrt(1 - mu2**2)
                    kpp_vecs = kpp * np.array([sin_theta2 * cos_phis, sin_theta2 * sin_phis, mu2 * np.ones_like(phis)]).T
                    T_vals = np.array([tri_func(k_vec, kp_vec, kpp_vec) for kpp_vec in kpp_vecs])
                    
                    integral_phi = np.trapz(T_vals * cos_phis, x=phis)
                    integrand_mu2.append(integral_phi)
                
                integral_mu2 = np.trapz(np.array(integrand_mu2) * np.sqrt(1 - mupps**2), x=mupps)
                integrand_mu1.append(integral_mu2)
            
            integral_mu1 = np.trapz(np.array(integrand_mu1) * np.sqrt(1 - mups**2), x=mups)
            integrand_kpp.append(integral_mu1)
        
        integral_kpp = np.trapz(np.array(integrand_kpp) * kpps**2, x=np.log(kpps))
        #print("    ", integral_kpp)
        integrand_kp.append(integral_kpp)
    
    result = np.trapz(np.array(integrand_kp) * kps**2, x=np.log(kps))
    return prefactor * result

#%%

def P_T_par(k_mag, aHf, tri_func, P_L_interp):
    k_vec = np.array([0, 0, k_mag])
    prefactor = aHf**2 / (2*np.pi)**5

    integrand_kp = []
    for i, kp in enumerate(kps):
        integrand_kpp = []
        for j, kpp in enumerate(kpps):
            integrand_mu1 = []
            for mu1 in mups:
                sin_theta1 = np.sqrt(1 - mu1**2)
                kp_vec = kp * np.array([sin_theta1, 0.0, mu1])
                
                integrand_mu2 = []
                for mu2 in mupps:
                    sin_theta2 = np.sqrt(1 - mu2**2)
                    kpp_vecs = kpp * np.array([sin_theta2 * cos_phis, sin_theta2 * sin_phis, mu2 * np.ones_like(phis)]).T
                    T_vals = np.array([tri_func(k_vec, kp_vec, kpp_vec) for kpp_vec in kpp_vecs])
                    
                    integral_phi = np.trapz(T_vals, x=phis)
                    integrand_mu2.append(integral_phi)
                    
                integral_mu2 = np.trapz(np.array(integrand_mu2) * mupps, x=mupps)
                integrand_mu1.append(integral_mu2)
            
            integral_mu1 = np.trapz(np.array(integrand_mu1) * mups, x=mups)
            integrand_kpp.append(integral_mu1)
            
        integral_kpp = np.trapz(np.array(integrand_kpp) * kpps**2, x=np.log(kpps))
        integrand_kp.append(integral_kpp)
    
    result = np.trapz(np.array(integrand_kp) * kps**2, x=np.log(kps))
    return prefactor * result

#%%

kmin = 1E-3
kmax = 1E1
ks = np.geomspace(kmin, kmax, n_k)
logk = np.log(ks)

Pk_cng = []

for ik, k in enumerate(ks):
    print(ik)
    t = P_T_perp(k, aHf, tri_4h, P_L_interp)
    print(t)
    Pk_cng.append(t)
    
Pk_cng = np.array(Pk_cng)

print('k_vals=', ks)
print("Pk_cng=", Pk_cng)

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

def tri_1h(k, kp, kpp):
    k1 = k - kp
    k2 = -(k - kpp)
    k3 = kp
    k4 = -kpp
    ks = [k1, k2, k3, k4]
    k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
    log10M = np.log10(M_vals)
    u1 = interp_pG.ev(np.log10(k_norms[0]), log10M)
    u2 = interp_pE.ev(np.log10(k_norms[1]), log10M)
    u3 = interp_pM.ev(np.log10(k_norms[2]), log10M)
    u4 = interp_pM.ev(np.log10(k_norms[3]), log10M)
    integrand = nM_vals * bM_vals * u1 * u2 * u3 * u4
    result = np.trapz(integrand, log10M)
    return result

#%%

kmin = 1E-3
kmax = 1E1
ks = np.geomspace(kmin, kmax, n_k)
logk = np.log(ks)

Pk_cng = []

for ik, k in enumerate(ks):
    print(ik)
    t = P_T_par(k, aHf, tri_1h, P_L_interp)
    print(t)
    Pk_cng.append(t)
    
Pk_cng = np.array(Pk_cng)

print('k_vals=', ks)
print("Pk_cng=", Pk_cng)
