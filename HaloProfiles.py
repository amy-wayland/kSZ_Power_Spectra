import numpy as np
import pyccl as ccl
from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator

#%%

class HaloProfileDensityBattaglia(ccl.halos.HaloProfile):
    '''
    Gas density profile from Battaglia 2016
    
    '''
    def __init__(self, *, mass_def,
                 rho0_A=4e3,
                 rho0_aM=0.29,
                 rho0_az=-0.66,
                 alpha_A=0.88,
                 alpha_aM=-0.03,
                 alpha_az=0.19,
                 beta_A=3.83,
                 beta_aM=0.04,
                 beta_az=-0.025,
                 gamma=-0.2,
                 xc=0.5,
                 alpha_interp_spacing=0.1,
                 beta_interp_spacing=0.3,
                 qrange=(1e-3, 1e3),
                 nq=128,
                 x_out=np.inf,
                 kind="rho_gas"):
        self.rho0_A = rho0_A
        self.rho0_aM = rho0_aM
        self.rho0_az = rho0_az
        self.alpha_A = alpha_A
        self.alpha_aM = alpha_aM
        self.alpha_az = alpha_az
        self.beta_A = beta_A
        self.beta_aM = beta_aM
        self.beta_az = beta_az
        self.gamma = gamma
        self.xc = xc
        self.kind = kind

        self.alpha_interp_spacing = alpha_interp_spacing
        self.beta_interp_spacing = beta_interp_spacing
        self.qrange = qrange
        self.nq = nq
        self.x_out = x_out
        self._fourier_interp = None
        super().__init__(mass_def=mass_def)


    def _AMz(self, M, a, A, aM, az):
        return A * (M * 1e-14) ** aM / a**az

    def _alpha(self, M, a):
        return self._AMz(M, a, self.alpha_A, self.alpha_aM, self.alpha_az)

    def _beta(self, M, a):
        return self._AMz(M, a, self.beta_A, self.beta_aM, self.beta_az)

    def _rho0(self, M, a):
        return self._AMz(M, a, self.rho0_A, self.rho0_aM, self.rho0_az)


    def update_parameters(self,
                          rho0_A=None,
                          rho0_aM=None,
                          rho0_az=None,
                          alpha_A=None,
                          alpha_aM=None,
                          alpha_az=None,
                          beta_A=None,
                          beta_aM=None,
                          beta_az=None,
                          gamma=None,
                          xc=None):
        if rho0_A is not None:
            self.rho0_A = rho0_A
        if rho0_aM is not None:
            self.rho0_aM = rho0_aM
        if rho0_az is not None:
            self.rho0_az = rho0_az
        if beta_A is not None:
            self.beta_A = beta_A
        if beta_aM is not None:
            self.beta_aM = beta_aM
        if beta_az is not None:
            self.beta_az = beta_az
        if alpha_A is not None:
            self.alpha_A = alpha_A
        if alpha_aM is not None:
            self.alpha_aM = alpha_aM
        if alpha_az is not None:
            self.alpha_az = alpha_az

        if xc is not None:
            self.xc = xc
        re_fourier=False
        if gamma is not None:
            if gamma != self.gamma:
                re_fourier = True
            self.gamma = gamma

        if re_fourier and (self._fourier_interp is not None):
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()


    def _form_factor(self, x, alpha, beta):
        return x**self.gamma / (1 + x**alpha) ** ((beta + self.gamma) / alpha)


    def _integ_interp(self):
        qs = np.geomspace(self.qrange[0], self.qrange[1], self.nq + 1)

        def integrand(x, alpha, beta):
            return self._form_factor(x, alpha, beta) * x

        alpha0 = self._alpha(1e15, 1.0) - self.alpha_interp_spacing
        alpha1 = self._alpha(1e10, 1 / (1 + 6.0)) + self.alpha_interp_spacing
        nalpha = int((alpha1 - alpha0) / self.alpha_interp_spacing)
        alphas = np.linspace(alpha0, alpha1, nalpha)
        beta0 = self._beta(1e10, 1.0) - 1
        beta1 = self._beta(1e15, 1 / (1 + 6.0)) + 1
        nbeta = int((beta1 - beta0) / self.beta_interp_spacing)
        betas = np.linspace(beta0, beta1, nbeta)
        f_arr = np.array(
            [
                [
                    [
                        quad(
                            integrand,
                            args=(
                                alpha,
                                beta,
                            ),
                            a=1e-4,
                            b=self.x_out,  # limits of integration
                            weight="sin",  # fourier sine weight
                            wvar=q,
                        )[0]
                        / q
                        for alpha in alphas
                    ]
                    for beta in betas
                ]
                for q in qs
            ]
        )
        f_arr[-1, :, :] = 1e-100
        Fqb = RegularGridInterpolator(
            [np.log(qs), betas, alphas],
            np.log(f_arr),
            fill_value=None,
            bounds_error=False,
            method="cubic",
        )
        return Fqb


    def _norm(self, cosmo, M, a):
        rho_c = ccl.rho_x(cosmo, a, self.mass_def.rho_type)
        fb = cosmo["Omega_b"] / (cosmo["Omega_b"] + cosmo["Omega_c"])
        rho0 = self._rho0(M, a) * rho_c * fb
        return rho0


    def _fourier(self, cosmo, k, M, a):
        if self._fourier_interp is None:
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()

        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        xrDelta = self.xc * self.mass_def.get_radius(cosmo, M_use, a) / a

        qs = k_use[None, :] * xrDelta[:, None]
        alphas = self._alpha(M_use, a)
        betas = self._beta(M_use, a)
        nk = len(k_use)
        ev = np.array(
            [
                np.log(qs).flatten(),
                (np.ones(nk)[None, :] * betas[:, None]).flatten(),
                (np.ones(nk)[None, :] * alphas[:, None]).flatten(),
            ]
        ).T
        ff = self._fourier_interp(ev).reshape([-1, nk])
        ff = np.exp(ff)
        nn = self._norm(cosmo, M_use, a)

        prof = (4 * np.pi * xrDelta**3 * nn)[:, None] * ff

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        xrDelta = self.xc * self.mass_def.get_radius(cosmo, M_use, a) / a
        alphas = self._alpha(M_use, a)
        betas = self._beta(M_use, a)

        nn = self._norm(cosmo, M_use, a)
        prof = self._form_factor(
            r_use[None, :] / xrDelta[:, None], alphas[:, None], betas[:, None]
        )
        prof *= nn[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
    
    
    def get_normalization(self, cosmo, a, *, hmc):

        def integrand(M):
            return self._fourier(cosmo=cosmo, k=1e-4, M=M, a=a)
        
        return hmc.integrate_over_massfunc(integrand, cosmo, a)


#%%    

class HaloProfileDensityHE(ccl.halos.HaloProfile):

    def __init__(self, *, mass_def, concentration,
                 lMc=14.0, beta=0.6, gamma=1.17,
                 gamma_T=1.0,
                 A_star=0.03, sigma_star=1.2,
                 eta_b=0.5):

        self.lMc = lMc
        self.beta = beta
        self.gamma = gamma
        self.A_star = A_star
        self.eta_b = eta_b
        self.sigma_star = sigma_star
        self._fourier_interp = None
        
        super().__init__(mass_def=mass_def, concentration=concentration)
    
    
    def _get_fractions(self, cosmo, M):
        fb = cosmo["Omega_b"] / (cosmo["Omega_b"] + cosmo["Omega_c"])
        Mbeta = (cosmo['h']*M*10**(-self.lMc))**self.beta
        f_star = self.A_star * \
            np.exp(-0.5*((np.log10(cosmo["h"]*M)-12.5)/self.sigma_star)**2)
        f_bound = (fb - f_star)*Mbeta/(1+Mbeta)
        f_ejected = fb - f_bound - f_star
        return f_bound, f_ejected, f_star
    
    
    def _form_factor(self, x):
        G = 1/(self.gamma - 1)
        return (np.log(1+x)/x)**G
        
    
    def _integ_interp(self):
        qs = np.geomspace(1e-2, 1e2, 128)

        def integrand(x):
            return self._form_factor(x) * x
        
        f_arr = np.array(
            [quad(integrand, 1e-4, 100, weight="sin", wvar=q)[0] / q
             for q in qs])
        
        f_arr[-1] = 1e-100
        
        Fqg = interp1d(np.log(qs), np.log(f_arr),
            bounds_error=False, fill_value='extrapolate', kind='linear')
        
        return Fqg
    
    
    def _norm_bound(self):
        
        def integrand(x): 
            return self._form_factor(x) * x**2
        
        return quad(integrand, 0, np.inf)[0]
        
    
    def _Ub_fourier(self, cosmo, k, M, a):
        
        if self._fourier_interp is None:
            with ccl.UnlockInstance(self):
                self._fourier_interp = self._integ_interp()
        
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
        
        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        cM = self.concentration(cosmo, M_use, a)
        r_s = rDelta/cM
        qs = k_use[None, :] * r_s[:, None]
        
        ff = self._fourier_interp(np.log(qs))
        ff = np.exp(ff)
        nn = self._norm_bound()
        prof = ff / nn
        
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof
    
    
    def _Ue_fourier(self, cosmo, k, M, a):
        M_use = np.atleast_1d(M)
        Delta = self.mass_def.get_Delta(cosmo, a)
        r_esc = 0.5 * np.sqrt(Delta) * self.mass_def.get_radius(cosmo, M_use, a)
        r_ej =  0.75 * self.eta_b * r_esc
        return np.exp(-0.5 * (k * r_ej)**2)
    
    
    def _fourier(self, cosmo, k, M, a):
        '''
        Computes the Fourier-space profile

        '''
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)
                
        Ub_k = np.array([self._Ub_fourier(cosmo, k_use, m_i, a) for m_i in M_use])
        Ue_k = np.array([self._Ue_fourier(cosmo, k_use, m_i, a) for m_i in M_use])
        
        fb = self._get_fractions(cosmo, M_use)[0]
        fe = self._get_fractions(cosmo, M_use)[1]
        
        prof_k = M_use[:, None] * (fb[:, None] * Ub_k + fe[:, None] * Ue_k) / a**3
        
        if np.ndim(k) == 0:
            prof_k = prof_k[:, 0]
        if np.ndim(M) == 0:
            prof_k = prof_k[0]

        return prof_k


    def _real(self, cosmo, r, M, a):
        '''
        Computes the real-space profile

        '''
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        rDelta = self.mass_def.get_radius(cosmo, M_use, a) / a
        cM = self.concentration(cosmo, M_use, a)
        r_s = rDelta/cM
        Delta = self.mass_def.get_Delta(cosmo, a)
        r_esc = 0.5 * np.sqrt(Delta) * self.mass_def.get_radius(cosmo, M_use, a)
        r_ej =  0.75 * self.eta_b * r_esc
        
        x = r_use[None, :] / r_s[:, None]
        norm = self._norm_bound()
        
        fb = self._get_fractions(cosmo, M_use)[0]
        fe = self._get_fractions(cosmo, M_use)[1]
        
        rho_b = fb * M_use[:, None] / (4 * np.pi * a**3 * r_s[:, None]**3 * norm) * self._form_factor(x)
        rho_e = fe * M_use[:, None] / (2 * np.pi * a**3 * r_ej[:, None]**2)**1.5 * \
                np.exp(-0.5 * (r[None, :] / r_ej[:, None])**2)
        
        prof = rho_b + rho_e
        
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return prof
    
    
    def get_normalization(self, cosmo, a, *, hmc):
        '''
        Calculates the physical gas density at scale factor a
        
        '''
        def rho_gas_integrand(M):
            fb, fe, fs = self._get_fractions(cosmo, M)
            return (fb + fe) * M / a**3
        
        return hmc.integrate_over_massfunc(rho_gas_integrand, cosmo, a)
    
