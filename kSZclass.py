import numpy as np
import pyccl as ccl
from scipy.special import erf

#%%

class kSZclass:
    
    def __init__(self, cosmo, k_arr, a_arr, pk_mm, pk_eg, pk_gm, pk_em, pk_ee, pk_gg):
        '''
        Class to calculate the 3D and angular power spectra for the kSZ effect
        
        Parameters
        ----------
        cosmo: pyccl cosmology object
        k_arr: numpy array of wavenumbers
        a_arr: numpy array of scale factors
        pk_mm: matter-matter pk2d object
        pk_eg: electron-galaxy pk2d object
        pk_gm: galaxy-matter pk2d object
        pk_em: electron-matter pk2d object
        pk_ee: electron-electron pk2d object
        pk_gg: galaxy-galaxy pk2d object
        
        '''
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
        '''
        Calculates the pk_mm and pk_ab convolution for the perpendicular 
        density-weighted peculiar velocity, q = (1+delta)v, mode
        where a = {e,g} and b = {e,g}
        
        '''
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
        '''
        Calculates the pk_am and pk_bm convolution for the perpendicular
        density-weighted peculiar velocity, q = (1+delta)v, mode
        where a = {e,g} and b = {e,g}
        
        '''
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
        '''
        Calculates the pk_mm and pk_ab convolution for the parallel
        density-weighted peculiar velocity, q = (1+delta)v, mode
        where a = {e,g} and b = {e,g}
        
        '''
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
        '''
        Calculates the pk_am and pk_bm convolution for the parallel
        density-weighted peculiar velocity, q = (1+delta)v, mode
        where a = {e,g} and b = {e,g}

        '''
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
    
    
    def _get_tracers(self, kind):
        '''
        Computes the tracers for galaxies and the kSZ effect

        Parameters
        ----------
        kind: kind of density-weighted peculiar velocity mode
              can be either "perp" or "par"

        '''
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
            
    
    def _get_pk2d(self, kind, ab):
        '''
        Computes pk2d objects for the full array of k and a values
        using the integrals P_perp_i or P_par_i where i = {1,2}

        Parameters
        ----------
        kind: kind of density-weighted peculiar velocity mode
              can be either "perp" or "par"
        ab: cross-correlation type
            can be either 'eg', 'gg', or 'ee'

        Returns
        -------
        pk1: 3D power spectrum from the pk_mm and pk_ab convolution
        pk2: 3D power spectrum from the pk_am and pk_bm convolution

        '''
        if kind == 'perp':
            P1 = self.P_perp_1
            P2 = self.P_perp_2
            
        elif kind == 'par':
            P1 = self.P_par_1
            P2 = self.P_par_2
            
        else:
            raise ValueError(f"Unknown power spectrum type {kind}")
            
        if ab == 'eg':
            pk_ab = self._pk_eg
            pk_am = self._pk_gm
            pk_bm = self._pk_em
        
        elif ab == 'gg':
            pk_ab = self._pk_gg
            pk_am = pk_bm = self._pk_gm
            
        elif ab == 'ee':
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
        '''
        Calculates the angular power spectrum using the 3D power spectrum
        from _get_pk2d() and the tracer objects from _get_tracers()

        Parameters
        ----------
        pk1: 3D power spectrum from the pk_mm and pk_ab convolution
        pk2: 3D power spectrum from the pk_am and pk_bm convolution
        ells: array of angular multipoles
        kind: kind of density-weighted peculiar velocity mode
              can be either "perp" or "par"
        ab: cross-correlation type
            can be either 'eg', 'gg', or 'ee'

        Returns
        -------
        Cl1: angular power spectrum from the pk_mm and pk_ab convolution
        Cl2: angular power spectrum from the pk_am and pk_bm convolution

        '''        
        tg, tk = self._get_tracers(kind)
        
        if kind == 'perp':
            pk1, pk2 = self._get_pk2d(kind, ab)
            prefac = ells * (ells+1) / (ells+0.5)**2
        
        elif kind == 'par':
            pk1, pk2 = self._get_pk2d(kind, ab)
            prefac = 1.0
          
        else:
            raise ValueError(f"Unknown power spectrum type {kind}")
            
        if ab == 'eg':
            ta = tg
            tb = tk
            
        elif ab == 'gg':
            ta = tb = tg
            
        elif ab == 'ee':
            ta = tb = tk
            
        else:
            raise ValueError(f"Unknown cross-correlation type {ab}")
            
        Cl1 = prefac * ccl.angular_cl(self._cosmo, ta, tb, ells, p_of_k_a=pk1)
        Cl2 = prefac * ccl.angular_cl(self._cosmo, ta, tb, ells, p_of_k_a=pk2)
        
        return Cl1, Cl2
    
    
    def get_Dl(self, ells, Cl):
        '''
        Converts C_ells into D_ells

        '''
        return ells * (ells + 1) * Cl / (2 * np.pi)
        
    
#%%
        
class Satellites:
    
    def __init__(self, M, M0, M1, M_min, nM):
        '''
        Class to calculate the contribution from central and satellite galaxies
        based on the HOD model from Zheng et al. 2005
        
        Parameters
        ----------
        M: halo mass
        M0: minimum mass of haloes that can host satellite galaxies
        M1: mass of haloes that on average contain one satellite galaxy
        M_min: minimum mass of haloes that can host central galaxies
        nM: halo mass function
            
        '''
        self._M = M
        self._M0 = M0
        self._M1 = M1
        self._M_min = M_min
        self._nM = nM
        
        
    def _N_c(self):
        '''
        Returns the mean number of central galaxies

        '''
        sig_lnM = 0.4 # Characteristic transition width
        return 0.5 * (1 + erf((np.log10(self._M / self._M_min)) / sig_lnM))


    def _N_s(self, alpha=1.0):
        '''
        Returns the mean number of satellite galaxies
        
        '''
        return np.heaviside(self._M - self._M0, 0) * ((self._M - self._M0) / self._M1)**(alpha)


    def mean_halo_mass(self):
        '''
        Returns the mean halo mass
        
        '''
        log10_M = np.log10(self._M)
        N_g = self._N_c() + self._N_s()
        i1 = self._M * self._nM * N_g
        i2 = self._nM * N_g
        return np.trapz(i1, log10_M) / np.trapz(i2, log10_M)


#%%

class HigherOrder:
    
    def __init__(self, cosmo, k_arr, a, M_vals, nM_vals, bM_vals, interp_pM, interp_pG, interp_pE, interp_P_L):
        
        self._cosmo = cosmo
        self._k_arr = k_arr
        self._a = a
        self._M_vals = M_vals
        self._nM_vals = nM_vals
        self._bM_vals = bM_vals
        self._interp_pM = interp_pM
        self._interp_pG = interp_pG
        self._interp_pE = interp_pE
        self._interp_P_L = interp_P_L
        
        self._H = cosmo['h'] * ccl.h_over_h0(self._cosmo, self._a) / ccl.physical_constants.CLIGHT_HMPC
        self._f = cosmo.growth_rate(self._a)
        self._aHf = self._a * self._H * self._f
        self._log10M = np.log10(self._M_vals)
    
    
    def _get_profiles(self, k_val):
        log10k = np.log10(k_val)
        uM = self._interp_pM.ev(log10k, self._log10M)
        uG = self._interp_pG.ev(log10k, self._log10M)
        uE = self._interp_pE.ev(log10k, self._log10M)
        return uM, uG, uE
    
        
    def _P_L(self, k_vec):
        k_norm = np.clip(np.linalg.norm(k_vec), 1e-4, 50)
        return max(self._interp_P_L(k_norm), 0.0)
    
    
    def _F2(self, k1, k2):
        
        k1_norm = np.linalg.norm(k1)
        k2_norm = np.linalg.norm(k2)
        dot_prod = np.dot(k1, k2)
        mu_12 = dot_prod / (k1_norm * k2_norm)
        
        if k1_norm < 1e-12 or k2_norm < 1e-12:
            return 0.0
    
        return 5/7 + (mu_12 / 2) * ((k1_norm / k2_norm) + (k2_norm / k1_norm)) + (2/7) * mu_12**2


    def _G2(self, k1, k2):
        
        k1_norm = np.linalg.norm(k1)
        k2_norm = np.linalg.norm(k2)
        dot_prod = np.dot(k1, k2)
        mu_12 = dot_prod / (k1_norm * k2_norm)
                
        if k1_norm < 1e-12 or k2_norm < 1e-12:
            return 0.0
        
        return 3/7 + (mu_12/2) * ((k1_norm / k2_norm) + (k2_norm / k1_norm)) + (4/7) * mu_12**2
        
    
    def _Q(self, k1, k2, k3):
        
        k123 = k1 + k2 + k3
        k23 = k2 + k3
        k1_norm = np.linalg.norm(k1)
        k123_norm = np.linalg.norm(k123)
        k23_norm = np.linalg.norm(k23)
        
                
        if k1_norm < 1e-12 or k23_norm < 1e-12:
            return 0.0
        
        prefact1 = (7 * np.dot(k123, k1)) / k1_norm**2
        term1 = prefact1 * self._F2(k2, k3)
        prefact2 = (7 * np.dot(k123, k23)) / k23_norm**2 + (2 * k123_norm**2 * np.dot(k23, k1)) / (k23_norm**2 * k1_norm**2)
        term2 = prefact2 * self._G2(k2, k3)
        
        return term1 + term2
    
    
    def _F3(self, k1, k2, k3):
        return (1/54) * (self._Q(k1, k2, k3) + self._Q(k2, k3, k1) + self._Q(k3, k1, k2))

    
    def _T_1122(self, k, kp, kpp):
        
        k1 = k - kp
        k2 = -(k - kpp)
        k3 = kp
        k4 = -kpp
        ks = [k1, k2, k3, k4]
        
        total = 0.0
        indices = [
            (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2),
            (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 3, 0), (2, 3, 1)]

        for i, j, k in indices:
            q_vec = ks[i] + ks[k]
            F2_1 = self._F2(q_vec, -ks[i])
            F2_2 = self._F2(q_vec, ks[j])
            P1 = self._P_L(q_vec)
            P2 = self._P_L(ks[i])
            P3 = self._P_L(ks[j])
            total += F2_1 * F2_2 * P1 * P2 * P3

        return 4 * total
    
    
    def _T_1113(self, k, kp, kpp):
        
        k1 = k - kp
        k2 = -(k - kpp)
        k3 = kp
        k4 = -kpp
        ks = [k1, k2, k3, k4]

        total = 0.0
        indices = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]

        for i, j, l in indices:
            F3_val = self._F3(ks[i], ks[j], ks[l])
            P1 = self._P_L(ks[i],)
            P2 = self._P_L(ks[j])
            P3 = self._P_L(ks[l])
            total += F3_val * P1 * P2 * P3

        return 6 * total
    
    
    def _T_tree_level(self, k, kp, kpp):
        return self._T_1122(k, kp, kpp) + self._T_1113(k, kp, kpp)
    
    
    def _B_tree_level(self, k, kp):
        
        k1 = k
        k2 = -kp
        k3 = -(k-kp)
        
        B_13 = 2 * self._F2(k1, k2) * self._P_L(k1) * self._P_L(k2)
        B_23 = 2 * self._F2(k2, k3) * self._P_L(k2) * self._P_L(k3)
        B_31 = 2 * self._F2(k3, k1) * self._P_L(k3) * self._P_L(k1)
        
        return B_13 + B_23 + B_31
    
    
    def _T_1h(self, k, kp, kpp):
        
        k1 = k - kp
        k2 = -(k - kpp)
        k3 = kp
        k4 = -kpp
        
        ks = [k1, k2, k3, k4]
        k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
        
        u1 = self._get_profiles(k_norms[0])[1]
        u2 = self._get_profiles(k_norms[1])[2] 
        u3 = self._get_profiles(k_norms[2])[0]
        u4 = self._get_profiles(k_norms[3])[0]

        integrand = self._nM_vals * self._bM_vals * u1 * u2 * u3 * u4

        return np.trapz(integrand, self._log10M)
    
    
    def _T_4h(self, k, kp, kpp):
        
        k1 = k - kp
        k2 = -(k - kpp)
        k3 = kp
        k4 = -kpp
        
        ks = [k1, k2, k3, k4]
        k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]
        
        u1 = self._get_profiles(k_norms[0])[1]
        u2 = self._get_profiles(k_norms[1])[2] 
        u3 = self._get_profiles(k_norms[2])[0]
        u4 = self._get_profiles(k_norms[3])[0]
        
        integrand1 = self._nM_vals * self._bM_vals * u1
        integrand2 = self._nM_vals * self._bM_vals * u2
        integrand3 = self._nM_vals * self._bM_vals * u3
        integrand4 = self._nM_vals * self._bM_vals * u4

        I1 = np.trapz(integrand1, self._log10M)
        I2 = np.trapz(integrand2, self._log10M)
        I3 = np.trapz(integrand3, self._log10M)
        I4 = np.trapz(integrand4, self._log10M)

        T_tree = self._T_tree_level(k, kp, kpp)

        return T_tree * I1 * I2 * I3 * I4
    
    
    def _B_1h(self, k, kp, density_kind):
        
        k1 = k
        k2 = -kp
        k3 = - (k - kp)
        ks = [k1, k2, k3]
        
        k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]

        u1 = self._get_profiles(k_norms[0])[0]
        u2 = self._get_profiles(k_norms[1])[0]
        
        if density_kind == 'galaxy':    
            u3 = self._get_profiles(k_norms[2])[1]
            
        elif density_kind == 'electron':
            u3 = self._get_profiles(k_norms[2])[2]
            
        else:
            raise ValueError(f"Unknown overdensity type {density_kind}")
        
        integrand = self._nM_vals * self._bM_vals * u1 * u2 * u3

        return np.trapz(integrand, self._log10M)
    
    
    def _B_3h(self, k, kp, density_kind):
        
        k1 = k
        k2 = -kp
        k3 = - (k - kp)
        ks = [k1, k2, k3]
        
        k_norms = [max(np.linalg.norm(ki), 1e-4) for ki in ks]

        u1 = self._get_profiles(k_norms[0])[0]
        u2 = self._get_profiles(k_norms[1])[0]
        
        if density_kind == 'galaxy':    
            u3 = self._get_profiles(k_norms[2])[1]
            
        elif density_kind == 'electron':
            u3 = self._get_profiles(k_norms[2])[2]
            
        else:
            raise ValueError(f"Unknown overdensity type {density_kind}")
            
        integrand1 = self._nM_vals * self._bM_vals * u1
        integrand2 = self._nM_vals * self._bM_vals * u2
        integrand3 = self._nM_vals * self._bM_vals * u3

        I1 = np.trapz(integrand1, self._log10M)
        I2 = np.trapz(integrand2, self._log10M)
        I3 = np.trapz(integrand3, self._log10M)

        B_tree = self._B_tree_level(k, kp)
        
        return B_tree * I1 * I2 * I3
    
    
    def _P_tri(self, k_vec, tri_func, kind, nk=20, nmu=20, nphi=20):
        
        k_mag = np.linalg.norm(k_vec)
        if k_mag < 1e-5:
            return 0.0
        
        # Integration grids
        lk_min, lk_max = -3, 1
        lkps = np.linspace(lk_min, lk_max, nk)
        lkpps = np.linspace(lk_min, lk_max, nk)
        kps = 10**lkps
        kpps = 10**lkpps
        mu_primes = np.linspace(-0.99, 0.99, nmu)
        mupp_primes = np.linspace(-0.99, 0.99, nmu)
        phis = np.linspace(0, 2*np.pi, nphi)
        
        sin_theta1s = np.sqrt(1 - mu_primes**2)
        sin_theta2s = np.sqrt(1 - mupp_primes**2)
        cos_phis = np.cos(phis)
        sin_phis = np.sin(phis)

        dlk = (lk_max - lk_min) / (nk - 1)
        dmu = 2 / nmu
        dphi = (2*np.pi) / nphi

        prefact = self._aHf**2 / (2*np.pi)**5
        result = 0.0
        
        for i, kp in enumerate(kps):
            for j, kpp in enumerate(kpps):
                for l, mu1 in enumerate(mu_primes):
                    sin_theta1 = sin_theta1s[l]

                    # Fixed k' vector in the x-z plane (due to azimuthal symmetry)
                    kp_vec = kp * np.array([sin_theta1, 0.0, mu1])

                    for m, mu2 in enumerate(mupp_primes):
                        sin_theta2 = sin_theta2s[m]

                        # Vectorised k'' vectors over phi
                        kpp_vecs = kpp * np.stack([sin_theta2 * cos_phis, sin_theta2 * sin_phis, mu2 * np.ones_like(phis)], axis=1)

                        # Evaluate trispectrum for each k''
                        T_vals = np.array([tri_func(k_vec, kp_vec, kpp_vec) for kpp_vec in kpp_vecs])
                        
                        # Approximate the 5D integral as a weighted 5D sum of the integrand times the volume elements
                        if kind == 'perp':
                            block_sum = np.sum(kp * kpp * sin_theta1 * sin_theta2 * cos_phis * T_vals * kp * kpp * dlk * dlk * dmu * dmu * dphi)
                        
                        elif kind == 'par':
                            block_sum = np.sum(kp * kpp * mu1 * mu2 * T_vals * kp * kpp * dlk * dlk * dmu * dmu * dphi) / k_mag**2
                        
                        else:
                            raise ValueError(f"Unknown power spectrum type {kind}")
                        
                        result += block_sum

        return prefact * result
    
    
    def _P_bi(self, k_vec, bi_func, kind, density_kind, nk=20, nmu=20, nphi=20):
        
        k_mag = np.linalg.norm(k_vec)
        if k_mag < 1e-5:
            return 0.0
        
        # Integration grids
        lk_min, lk_max = -3, 1
        lkps = np.linspace(lk_min, lk_max, nk)
        kps = 10**lkps
        mu_primes = np.linspace(-0.99, 0.99, nmu)
        phis = np.linspace(0, 2 * np.pi, nphi)
        
        sin_thetas = np.sqrt(1 - mu_primes**2)
        cos_phis = np.cos(phis)
        sin_phis = np.sin(phis)

        dlk = (lk_max - lk_min) / (nk - 1)
        dmu = 2 / nmu
        dphi = (2*np.pi) / nphi

        prefact = self._aHf**2 / (2*np.pi)**3
        result = 0.0

        for i, kp in enumerate(kps):
            for m, mu in enumerate(mu_primes):
                sin_theta = sin_thetas[m]

                # k' vectors over phi
                kp_vecs = kp * np.stack([sin_theta * cos_phis, sin_theta * sin_phis, mu * np.ones_like(phis)], axis=1)

                # Evaluate bispectrum for each phi
                B_vals = np.array([bi_func(k_vec, kp_vec, density_kind) for kp_vec in kp_vecs])
                
                if kind == 'par':
                    block_sum = np.sum(k_mag * kp * mu * B_vals * kp * dlk * dmu * dphi) / k_mag**2
                
                else:
                    raise ValueError(f"Unknown power spectrum type {kind}")
                
                result += block_sum

        return prefact * result
        
    
    def compute_P(self, k, spectra_type, kind, term, density_kind=None):
        
        k_vec = np.array([0, 0, k])
        
        if spectra_type == 'bispectrum':          
            if term == '1h':
                P = self._P_bi(k_vec, self._B_1h, kind, density_kind)           
            elif term == '3h':
                P = self._P_bi(k_vec, self._B_3h, kind, density_kind)                
            else: 
                raise ValueError(f"Unknown halo term {term}")
                               
        elif spectra_type == 'trispectrum':           
            if term == '1h':
                P = self._P_tri(k_vec, self._T_1h, kind)            
            elif term == '4h':
                P = self._P_tri(k_vec, self._T_4h, kind)
            else:
                raise ValueError(f"Unknown halo term {term}")
                
        else:
            raise ValueError(f"Unknown spectra type {spectra_type} or kind {kind}")
        
        return P
    
    
    def compute_P_of_k(self, spectra_type, kind, term, density_kind=None):
        P_of_k = np.array([self.compute_P(k, spectra_type, kind, term, density_kind) for k in self._k_arr])
        return P_of_k
    
