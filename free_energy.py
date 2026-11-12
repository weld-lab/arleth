import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
from scipy.integrate import quad

from utils import *
from fixed_params import *

DEBUG = False

def free_energy(alpha):
    """ Level-cut parameters (eqn 4)
    """
    alpha_c = norm.ppf(1 - phi_w)
    if alpha < alpha_c:
        # alpha < beta : oil-in-water
        beta = norm.ppf(norm.cdf(alpha) + phi_w)
    elif alpha > alpha_c:
        # alpha > beta : water-in-oil
        beta = norm.ppf(norm.cdf(alpha) + phi_w - 1)
    else:
        raise ValueError("One-cut limit")


    """ [A] constant (eqn 6)
    The paper gives A = e^(-alpha^2) + e^(-beta^2)
    We think it's an error, and the actual value should be,
    A = np.exp(-alpha*alpha/2) + np.exp(-beta*beta/2)
    As expressed in Teubner1991
    """
    A = np.exp(-alpha*alpha/2) + np.exp(-beta*beta/2)


    """ We now have all the parameters to compute a (eqn 16)
    """
    a = (π*π * 15 * phi_s) / (16 * kappa * A*A)
    
    """ Let's solve for b, c (eqn 19, 20)
    """
    
    def spectral_eqns(params):
        b, c = params
        
        # nu(k) should stay positive over [0, kc]
        # maybe some additional tests should be done
        
        delta = np.sqrt(complex(b*b - 4*c))
        
        r_minus = np.sqrt(complex(b-delta))
        r_plus  = np.sqrt(complex(b+delta))
        
        T_minus = r_minus * np.arctanh(np.sqrt(2)*kc / r_minus)
        T_plus = r_plus * np.arctanh(np.sqrt(2)*kc / r_plus)
        
        F19 = ((T_minus - T_plus) / (np.sqrt(2)*delta)) - 2 * π**2 / a
        
        
        F20 = kc + np.sqrt(2)/(4*delta) * (
            (b - delta)*T_minus - (b + delta)*T_plus
        ) - 8*kappa*phi_s*π**2/5
        
        return [F19.real, F20.real]
    

    spectral_solution = least_squares(
        spectral_eqns,
        x0=[0.10, 0.005], # only a guess, might need to change it later
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=1500
    )


    residual = np.linalg.norm(spectral_solution.fun)

    if not spectral_solution.success or residual > 1e-8:
        raise RuntimeError(
            f"alpha={alpha:.6f}, beta={beta:.6f}, "
            f"b={spectral_solution.x[0]:.6f}, "
            f"c={spectral_solution.x[1]:.6f}, "
            f"residual={residual:.3e}, "
            f"message={spectral_solution.message}"
        )
        
    b,c = spectral_solution.x
    
    if DEBUG: print(f"{a = :.8f}")
    if DEBUG: print(f"{b = :.8f}")
    if DEBUG: print(f"{c = :.8f}")
    
    
    """ Arleth gives 2 constraints to verify the values aren't too bad
    <k^0> = 1 (eqn 9)
    <k^2> exists and is equal to a constant (eqn 10)
    """
    def spectral_moment(n):
        result, _ = quad(
            lambda k: k**(n+2) * a / (k**4 - b*k**2 + c),
            0, kc)
        
        return result/(2*π**2)
    
    k0 = spectral_moment(0)
    k2 = spectral_moment(2)
    
    k2_target = 3 * (π * phi_s / (2*A))**2
    
    if DEBUG: print("<k^0> =", k0, "  Target = 1.0")
    if DEBUG: print("<k^2> =", k2, "  Target = ", k2_target)
    
    
    k4 = spectral_moment(4)
    if DEBUG: print("<k^4>", k4)
    
    
    """ We can now compute the mean curvatures,
    <H> (eqn 12),
    <H^2> (eqn 13),
    <K> (eqn 14)
    
    for which we will need [B] & [B^2]
    """
    w_alpha = np.exp(-alpha**2 / 2)
    w_beta = np.exp(-beta**2 / 2)
    
    B = (alpha*w_alpha - beta*w_beta) / A
    B2 = (alpha**2*w_alpha + beta**2*w_beta) / A
    
    """Minus sign to match Arleth's
    figures and discussion which use the
    opposite convention to printed eqn 12.
    Reversing the normal changes H
    but leaves H2 and K unchanged.
    """ 
    H = - 0.5 * np.sqrt(π*k2/6) * B
    H2 = k2/6 * (B2 - 1) + k4/(5*k2)
    K = k2/6 * (B2 - 1)
    
    if DEBUG: print("<H> =", H)
    if DEBUG: print("<H^2> =", H2)
    if DEBUG: print("<K> =", K)
    
    

    """ Finally, the free energy term per unit of volume
    (here expressed in kT/nm^3, eqn 5)
    """
    
    integral, _ = quad(
        lambda k: k**2 * np.log(a / (k**4 - b*k**2 + c)),
        0, kc)
    f_entropy = -integral / (4*π**2) # 2π^2 ? 
    
    S_over_V = phi_s/ds
    f_H = S_over_V * ( 2*kappa*(H2 - 2*H0*H + H0**2) + kappa_bar*K )
    
    
    f_free = f_H + f_entropy
    
    if DEBUG: print(f"Free energy {f_free = :.8f} kT/nm^3")
    return f_free, beta, a, b, c
