
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import fixed_params as params

from free_energy import free_energy
from minimization import minimize_free_energy
from visualize import visualize_microemulsion
from structure_factor import structure_factor


""" Numerical parameters
"""
N = 128
dx = 1.0  # nm

low_bound = -10
high_bound = 10
eps = 1e-3


""" Minimize the free energy
"""
alpha_opt, beta_opt, a_opt, b_opt, c_opt, f_min = minimize_free_energy(
    params.phi_w, params.phi_s,
    params.kappa, params.kappa_bar, params.H0,
    low_bound=low_bound, high_bound=high_bound
)

print(f"Optimal alpha = {alpha_opt:.8f}")
print(f"Minimum free energy = {f_min:.8f} kT/nm^3")

print(f"alpha = {alpha_opt:.8f}")
print(f"beta  = {beta_opt:.8f}")
print(f"a     = {a_opt:.8f}")
print(f"b     = {b_opt:.8f}")
print(f"c     = {c_opt:.8f}")


""" Plot the free energy profile
"""
alpha_c = norm.ppf(1 - params.phi_w)

alphas_lower = np.linspace(low_bound, alpha_c - eps, 100)
alphas_upper = np.linspace(alpha_c + eps, high_bound, 100)


def energy_profile(alphas):
    energies = []

    for alpha in alphas:
        try:
            energy, *_ = free_energy(
                alpha,
                phi_w=params.phi_w,
                phi_s=params.phi_s,
                kappa=params.kappa,
                kappa_bar=params.kappa_bar,
                H0=params.H0,
                kc=params.kc,
                ds=params.ds
            )
            energies.append(energy)

        except RuntimeError:
            print(f"Failed for alpha = {alpha:.6f}")
            energies.append(np.nan)

    return np.array(energies)


energies_lower = energy_profile(alphas_lower)
energies_upper = energy_profile(alphas_upper)

plt.plot(alphas_lower, energies_lower, label=r"$\alpha < \alpha_c$", color="r")
plt.plot(alphas_upper, energies_upper, label=r"$\alpha > \alpha_c$", color="k")

plt.scatter(
    alpha_opt, f_min,
    marker="*",
    s=200,
    color="gold",
    edgecolor="black",
    zorder=5,
    label="minimum"
)

plt.axvline(
    alpha_c,
    color="gray",
    linestyle="--",
    label="One-cut limit"
)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$f_{\mathrm{free}}$ ($kT/\mathrm{nm}^3$)")
plt.legend()
plt.show()


""" Visualize the microemulsion structure
"""
field, water = visualize_microemulsion(
    alpha_opt, beta_opt,
    a_opt, b_opt, c_opt,
    N=N, dx=dx
)

print("Field mean =", field.mean())
print("Field variance =", field.var())
print("Water fraction =", water.mean(), " Target =", params.phi_w)


""" Calculate the structure factor
"""
q_values = np.linspace(0.02, 0.6, 200)

I = structure_factor(
    q_values,
    alpha_opt, beta_opt,
    a_opt, b_opt, c_opt
)

plt.plot(q_values, I)
plt.xlabel(r"$q$ (nm$^{-1}$)")
plt.ylabel(r"$I(q)$ (a.u.)")
plt.show()
