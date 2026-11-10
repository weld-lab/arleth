import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from utils import *
from fixed_params import *
from free_energy import free_energy
from visualize import visualize_microemulsion


""" Numerical parameters
""" 
N = 256 # number of grid points per dimension
dx = 1.0 # in nm, spacing of grid points
L = N*dx # in nm, simulation box length
V = L**3 # in nm^3, simulation box volume

low_bound = -10
high_bound = 2


""" Let's scan the free energy profile in function of alpha
"""
alpha_c = norm.ppf(1 - phi_w)
eps = 1e-3

def scan_energy(alphas):
    energies = []
    for alpha in alphas:
        try:
            energy, *_ = free_energy(alpha)
            energies.append(energy)
        except RuntimeError:
            print(f"Failed for alpha = {alpha:.6f}")
            energies.append(np.nan)
    return np.array(energies)

alphas_ow = np.linspace(low_bound, alpha_c - eps, 100)
alphas_wo = np.linspace(alpha_c + eps, high_bound, 100)

energies_ow = scan_energy(alphas_ow)
energies_wo = scan_energy(alphas_wo)


""" By now, we should be able to find the global minimum on the interval [low_bound, high_bound]
"""
def objective(alpha):
    energy, *_ = free_energy(alpha)
    return energy


def find_minima(alphas, energies):
    minima = []

    # Check the boundaries of the explored interval
    for i in [0, len(alphas) - 1]:
        if np.isfinite(energies[i]):
            minima.append((alphas[i], energies[i]))

    # Find and refine interior local minima
    for i in range(1, len(alphas) - 1):
        if not np.all(np.isfinite(energies[i-1:i+2])):
            continue
        if energies[i] < energies[i-1] and energies[i] < energies[i+1]:
            result = minimize_scalar(
                objective,
                bounds=(alphas[i-1], alphas[i+1]),
                method="bounded",
                options={"xatol": 1e-9}
            )
            if result.success:
                minima.append((result.x, result.fun))
    return minima


minima_ow = find_minima(alphas_ow, energies_ow)
minima_wo = find_minima(alphas_wo, energies_wo)

candidates = (
    [("O/W", alpha, energy) for alpha, energy in minima_ow]
    + [("W/O", alpha, energy) for alpha, energy in minima_wo]
)

branch, alpha_opt, f_min = min(candidates, key=lambda x: x[2])

print(f"Optimal branch = {branch}")
print(f"Optimal alpha = {alpha_opt:.8f}")
print(f"Minimum free energy = {f_min:.8f} kT/nm^3")


f_min, beta_opt, a_opt, b_opt, c_opt = free_energy(alpha_opt)

print(f"alpha = {alpha_opt:.8f}")
print(f"beta  = {beta_opt:.8f}")
print(f"a     = {a_opt:.8f}")
print(f"b     = {b_opt:.8f}")
print(f"c     = {c_opt:.8f}")


""" Plot
"""
plt.scatter(
    alpha_opt, f_min,
    marker="*",
    s=200,
    color="gold",
    edgecolor="black",
    zorder=5,
    label="minimum"
)


plt.plot(alphas_ow, energies_ow, label="O/W", color='r')
plt.plot(alphas_wo, energies_wo, label="W/O", color='k')

plt.axvline(alpha_c, color="gray", linestyle="--",
            label="One-cut limit")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$f_{\mathrm{free}}$ ($k T/\mathrm{nm}^3$)")
plt.legend()
plt.show()

""" Visualize the microemulsion structure
"""

field, water = visualize_microemulsion(
    alpha_opt, beta_opt,
    a_opt, b_opt, c_opt,
    N=128, dx=dx
)


print("Field mean =", field.mean())
print("Field variance =", field.var())
print("Water fraction =", water.mean(), " Target = ", phi_w)

