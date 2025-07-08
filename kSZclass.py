import numpy as np
import pyccl as ccl
from scipy.special import erf

#%%

class kSZclass:
    
    def __init__(self, cosmo, k_arr, a_arr, pk_mm, pk_eg, pk_gm, pk_em, pk_ee, pk_gg):

        self._cosmo = cosmo
        self._k_arr = k_arr
        self._a_arr = a_arr
        self._pk_mm = pk_mm
        self._pk_eg = pk_eg
        self._pk_gm = pk_gm
        self._pk_em = pk_em
        self._pk_ee = pk_ee
        self._pk_gg = pk_gg

        self._lk_arr = np.log(self._k_arr)
        self._z_arr = (1/self._a_arr) - 1
        self._H_arr = self._cosmo['h'] * ccl.h_over_h0(self._cosmo, self._a_arr) / ccl.physical_constants.CLIGHT_HMPC
        self._f_arr = self._cosmo.growth_rate(self._a_arr)
        self._aHf_arr = self._a_arr * self._H_arr * self._f_arr

         
    def P_perp_1(self, k, a, a_index, pk_ab):
        
        aHf = self._aHf_arr[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 128)
        lk_vals = np.log(np.logspace(-4, 1, 128))

        def integrand2(mu, kp):
            q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
            return (1-mu**2) * pk_ab(q, a, self._cosmo)

        def integrand1(lkp):
            kp = np.exp(lkp)
            integrand = integrand2(mu_vals, kp)
            integral = np.trapz(integrand, mu_vals)
            return kp * integral * self._pk_mm(kp, a, self._cosmo)

        integrand = np.array([integrand1(lk) for lk in lk_vals])
        integral = np.trapz(integrand, lk_vals)
        
        return integral * aHf**2 / (2*np.pi)**2
    

    def P_perp_2(self, k, a, a_index, pk_am, pk_bm):
        
        aHf = self._aHf_arr[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 128)
        lk_vals = np.log(np.logspace(-4, 1, 128))
        
        def integrand2(mu, kp):
            q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
            return -(1-mu**2) * pk_am(q, a) / q**2
        
        def integrand1(lkp):
            kp = np.exp(lkp)
            integrand = integrand2(mu_vals, kp)
            integral = np.trapz(integrand, mu_vals)
            return kp**3 * integral * pk_bm(kp, a)

        integrand = np.array([integrand1(lk) for lk in lk_vals])
        integral = np.trapz(integrand, lk_vals)
        
        return integral * aHf**2 / (2*np.pi)**2
    
    
    def P_par_1(self, k, a, a_index, pk_ab):
        
        aHf = self._aHf_arr[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 128)
        lk_vals = np.log(np.logspace(-4, 1, 128))

        def integrand2(mu, kp):
            q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
            return mu**2 * pk_ab(q, a)

        def integrand1(lkp):
            kp = np.exp(lkp)
            integrand = integrand2(mu_vals, kp)
            integral = np.trapz(integrand, mu_vals)
            return kp * integral * self._pk_mm(kp, a)

        integrand = np.array([integrand1(lk) for lk in lk_vals])
        integral = np.trapz(integrand, lk_vals)
        
        return integral * aHf**2 / (2*np.pi)**2


    def P_par_2(self, k, a, a_index, pk_am, pk_bm):
        
        aHf = self._aHf_arr[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 128)
        lk_vals = np.log(np.logspace(-4, 1, 128))
        
        def integrand2(mu, kp):
            q = np.sqrt(k**2 + kp**2 - 2*k*kp*mu)
            return mu * (k-kp*mu) * pk_am(q, a) / q**2
        
        def integrand1(lkp):
            kp = np.exp(lkp)
            integrand = integrand2(mu_vals, kp)
            integral = np.trapz(integrand, mu_vals)
            return kp**2 * integral * pk_bm(kp, a)

        integrand = np.array([integrand1(lk) for lk in lk_vals])
        integral = np.trapz(integrand, lk_vals)
        
        return integral * aHf**2 / (2*np.pi)**2
    
    
    def get_tracers(self, kind):
        
        xH = 0.76
        sigmaT_over_mp = 8.30883107e-17
        ne_times_mp = 0.5 * (1+xH) * self._cosmo['Omega_b'] * self._cosmo['h']**2 * ccl.physical_constants.RHO_CRITICAL
        sigmaTne = ne_times_mp * sigmaT_over_mp

        nz = np.exp(-0.5 * ((self._z_arr - 0.55) / 0.05)**2) 
        
        sort_idx = np.argsort(self._z_arr)
        self._z_arr = self._z_arr[sort_idx]
        nz = nz[sort_idx]

        kernel_g = ccl.get_density_kernel(self._cosmo, dndz=(self._z_arr, nz))
        chis = ccl.comoving_radial_distance(self._cosmo, 1/(1+self._z_arr))
        
        tg = ccl.Tracer()
        tk = ccl.Tracer()
        
        if kind == 'perp': 
            tg.add_tracer(self._cosmo, kernel=kernel_g)
            tk.add_tracer(self._cosmo, kernel=(chis, sigmaTne/self._a_arr**2))
            
        elif kind == 'par':
            tg.add_tracer(self._cosmo, kernel=kernel_g, der_bessel=1)
            tk.add_tracer(self._cosmo, kernel=(chis, sigmaTne/self._a_arr**2), der_bessel=1)
        
        else:
            raise ValueError(f"Unknown power spectrum type {kind}")
            
        return tg, tk
            
    
    def get_pk2d(self, kind, ab):
        
        if kind == 'perp':
            P1 = self.P_perp_1
            P2 = self.P_perp_2
            
        elif kind == 'par':
            P1 = self.P_par_1
            P2 = self.P_par_2
            
        else:
            raise ValueError(f"Unknown power spectrum type {kind}")
            
        if ab == 'gk':
            pk_ab = self._pk_eg
            pk_am = self._pk_gm
            pk_bm = self._pk_em
        
        elif ab == 'gg':
            pk_ab = self._pk_gg
            pk_am = pk_bm = self._pk_gm
            
        elif ab == 'kk':
            pk_ab = self._pk_ee
            pk_am = pk_bm = self._pk_em
            
        else:
            raise ValueError(f"Unknown cross-correlation type {ab}")
        
        pk1 = np.zeros((len(self._k_arr), len(self._a_arr)))
        pk2 = np.zeros((len(self._k_arr), len(self._a_arr)))
        
        for i, a in enumerate(self._a_arr):
            p1 = np.array([P1(k, a, i, pk_ab) for k in self._k_arr])
            p2 = np.array([P2(k, a, i, pk_am, pk_bm) for k in self._k_arr])
            pk1[:, i] = p1
            pk2[:, i] = p2
        
        sort_idx = np.argsort(self._a_arr)
        self._a_arr = self._a_arr[sort_idx]
        self._H_arr = self._H_arr[sort_idx]
        self._f_arr = self._f_arr[sort_idx]
        
        pk1 = pk1[:, sort_idx]
        pk2 = pk2[:, sort_idx]
        
        pk1 = ccl.Pk2D(a_arr=self._a_arr, lk_arr=np.log(self._k_arr), pk_arr=pk1.T, is_logp=False)
        pk2 = ccl.Pk2D(a_arr=self._a_arr, lk_arr=np.log(self._k_arr), pk_arr=pk2.T, is_logp=False)
        
        return pk1, pk2
    
    
    def get_Cl(self, pk1, pk2, ells, kind, ab):
        
        tg, tk = self.get_tracers(kind)
        
        if kind == 'perp':
            pk1, pk2 = self.get_pk2d(kind, ab)
            prefac = ells * (ells+1) / (ells+0.5)**2
        
        elif kind == 'par':
            pk1, pk2 = self.get_pk2d(kind, ab)
            prefac = 1.0
          
        else:
            raise ValueError(f"Unknown power spectrum type {kind}")
            
        if ab == 'gk':
            ta = tg
            tb = tk
            
        elif ab == 'gg':
            ta = tb = tg
            
        elif ab == 'kk':
            ta = tb = tk
            
        else:
            raise ValueError(f"Unknown cross-correlation type {ab}")
            
        Cl1 = prefac * ccl.angular_cl(self._cosmo, ta, tb, ells, p_of_k_a=pk1)
        Cl2 = prefac * ccl.angular_cl(self._cosmo, ta, tb, ells, p_of_k_a=pk2)
        
        return Cl1, Cl2
    
    
    def get_Dl(self, ells, Cl):
        
        return ells * (ells + 1) * Cl / (2 * np.pi)
        
    
#%%
        
class Satellites:
    
    def __init__(self, M, M0, M1, M_min, nM):

        self._M = M
        self._M0 = M0
        self._M1 = M1
        self._M_min = M_min
        self._nM = nM
        
        
    def N_c(self):
        '''
        Returns the mean number of central galaxies

        '''
        sig_lnM = 0.4
        return 0.5 * (1 + erf((np.log10(self._M / self._M_min)) / sig_lnM))


    def N_s(self, alpha=1.0):
        '''
        Returns the mean number of satellite galaxies
        
        '''
        return np.heaviside(self._M - self._M0, 0) * ((self._M - self._M0) / self._M1)**(alpha)


    def mean_halo_mass(self):
        '''
        Returns the mean halo mass
        
        '''
        log10_M = np.log10(self._M)
        N_g = self.N_c() + self.N_s()
        i1 = self._M * self._nM * N_g
        i2 = self._nM * N_g
        return np.trapz(i1, log10_M) / np.trapz(i2, log10_M)
