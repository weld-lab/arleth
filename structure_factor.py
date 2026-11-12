import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, simpson

from fixed_params import *
from utils import *

def gaussian_correlation(r, a, b, c):
    """ Expression from Pieruschka1992
        for the gaussian correlation function g(r)
        simplified to the isotropic case
    """
    def integrand(k):
        nu = a / (k**4 - b*k**2 + c)
        return k**2 * nu * np.sinc(k*r/π) # for numpy, sinc(r) = sin(πr)/(πr)

    result, _ = quad(integrand, 0, kc)

    return result/(2*π**2)        


def clipped_correlation(r, alpha, beta,
                        a, b, c):
    """ Eqn 4 from Pieruschka1992
    """
    lo, hi = sorted([alpha, beta])
    g = gaussian_correlation(r, a, b, c)
    g = np.clip(g, -1.0, 1.0)

    gamma0 = norm.cdf(hi) - norm.cdf(lo)

    def integrand(theta):
        s = np.sin(theta)
        cos2 = np.cos(theta)**2

        term1 = np.exp(-lo**2 / (1 + s))
        term2 = np.exp(-hi**2 / (1 + s))
        term3 = 2 * np.exp(-0.5 * (lo**2 - 2*lo*hi*s + hi**2)/cos2)
        
        return term1+term2-term3

    integral, _ = quad(integrand, np.arcsin(g), π/2)

    return gamma0 - integral / (2*π)

def normalized_correlation(r, alpha, beta,
                           a, b, c):
    """ Eqn 5 from Pieruschka1992
    """
    lo, hi = sorted([alpha, beta])

    p = norm.cdf(hi) - norm.cdf(lo)
    gamma = clipped_correlation(r, alpha, beta, a, b, c)

    return (gamma - p**2) / (p * (1-p))

def structure_factor(alpha, beta,
                     a, b, c,
                     qs):
    from scipy.integrate import simpson


def structure_factor(q_values, alpha, beta, a, b, c,
                     r_max=300.0, n_r=1501):
    """ Structure factor from Pieruschka1992, eqn 6.
    """
    r = np.linspace(0, r_max, n_r)

    gamma = np.array([
        normalized_correlation(ri, alpha, beta, a, b, c)
        for ri in r
    ])

    lo, hi = sorted([alpha, beta])
    p = norm.cdf(hi) - norm.cdf(lo)

    intensities = []

    for q in np.atleast_1d(q_values):
        integrand = r**2 * gamma * np.sinc(q*r/np.pi)
        integral = simpson(integrand, x=r)

        intensities.append(4*np.pi*p*(1-p)*integral)

    return np.array(intensities)
