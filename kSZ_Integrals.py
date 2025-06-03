import numpy as np
import pyccl as ccl

#%%
# Class to calculate the integrals for the 3D power spectra for the transverse and longitduinal components

#%%

class kSZIntegral:
    
    def __init__(self, cosmo, k_arr, k_prime_arr, a_arr, H):

        self._cosmo = cosmo
        self._k_arr = k_arr
        self._k_prime_arr = k_prime_arr
        self._a_arr = a_arr
        self._H = H
        
        self._lk_arr = np.log(self._k_arr)
        self._a_dot = self._a_arr * self._H
        self._D = ccl.growth_factor(self._cosmo, self._a_arr)
        self._ln_a = np.log(self._a_arr)
        self._ln_D = np.log(self._D)
        self._f = np.gradient(self._ln_D, self._ln_a)
         
        
    def integral_perp_1(self, k, P_of_k_1, P_of_k_2, a_index, a):
        '''
        Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term for the perpendicular component

        '''
        a_dot_val = self._a_dot[a_index]
        f_val = self._f[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 1000)

        def integrand(mu, k_prime):
            q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
            return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * (1 - mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

        def int_over_mu(k_prime):
            vals = integrand(mu_vals, k_prime)
            return np.trapz(vals, mu_vals)

        integrand_k_prime = np.array([int_over_mu(k_p) for k_p in self._k_prime_arr])
        
        return np.trapz(integrand_k_prime, self._k_prime_arr)


    def integral_perp_2(self, k, P_of_k_1, P_of_k_2, a_index, a):
        '''
        Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term for the perpendicular component

        '''
        a_dot_val = self._a_dot[a_index]
        f_val = self._f[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 1000)
        
        def integrand(mu, k_prime):
            p = k**2 + k_prime**2 - 2 * k * k_prime * mu
            q = np.sqrt(p)
            return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * k_prime**2 * (1-mu**2) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / p
        
        def int_over_mu(k_prime):
            vals = integrand(mu_vals, k_prime)
            return np.trapz(vals, mu_vals)

        integrand_k_prime = np.array([int_over_mu(k_p) for k_p in self._k_prime_arr])
        
        return np.trapz(integrand_k_prime, self._k_prime_arr)
    
    
    def integral_par_1(self, k, P_of_k_1, P_of_k_2, a_index, a):
        '''
        Calculates the contribution from the < \delta_g \delta_e^* > < \delta_m \delta_m^* > term for the parallel component

        '''
        a_dot_val = self._a_dot[a_index]
        f_val = self._f[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 2000)

        def integrand(mu, k_prime):
            q = np.sqrt(k**2 + k_prime**2 - 2 * k * k_prime * mu)
            return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * mu**2 * P_of_k_1(k_prime, a) * P_of_k_2(q, a)

        def int_over_mu(k_prime):
            vals = integrand(mu_vals, k_prime)
            return np.trapz(vals, mu_vals)

        integrand_k_prime = np.array([int_over_mu(k_p) for k_p in self._k_prime_arr])
        
        return np.trapz(integrand_k_prime, self._k_prime_arr)


    def integral_par_2(self, k, P_of_k_1, P_of_k_2, a_index, a):
        '''
        Calculates the contribution from the < \delta_g \delta_m^* > < \delta_m \delta_e^* > term for the parallel component

        '''
        a_dot_val = self._a_dot[a_index]
        f_val = self._f[a_index]
        mu_vals = np.linspace(-0.99, 0.99, 2000)
        
        def integrand(mu, k_prime):
            #p = k**2 + k_prime**2 - 2 * k * k_prime * mu
            p = np.maximum(k**2 + k_prime**2 - 2 * k * k_prime * mu, 1e-4)
            q = np.sqrt(p)
            return (a_dot_val * f_val)**2 * (1/(2 * np.pi)**2) * mu * (k/k_prime - mu) * P_of_k_1(k_prime, a) * P_of_k_2(q, a) / (p + 1e-10)
        
        def int_over_mu(k_prime):
            vals = integrand(mu_vals, k_prime)
            return np.trapz(vals, mu_vals)

        integrand_k_prime = np.array([int_over_mu(k_p) for k_p in self._k_prime_arr])
        
        return np.trapz(integrand_k_prime, self._k_prime_arr)