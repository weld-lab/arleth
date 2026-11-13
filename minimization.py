import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from free_energy import free_energy
import fixed_params as params


def minimize_free_energy(phi_w=params.phi_w, phi_s=params.phi_s,
                         kappa=params.kappa, kappa_bar=params.kappa_bar,
                         H0=params.H0,
                         low_bound=-10, high_bound=10,
                         n_alpha=100, eps=1e-3):
    """Minimize the free energy with respect to alpha."""

    alpha_c = norm.ppf(1 - phi_w)

    def objective(alpha):
        energy, *_ = free_energy(
            alpha, phi_w, phi_s, kappa, kappa_bar, H0
        )
        return energy

    def scan_energy(alphas):
        energies = []

        for alpha in alphas:
            try:
                energies.append(objective(alpha))
            except RuntimeError:
                energies.append(np.nan)

        return np.array(energies)

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

                if result.success and np.isfinite(result.fun):
                    minima.append((result.x, result.fun))

        return minima

    # Scan both branches
    alphas_lower = np.linspace(
        low_bound, alpha_c - eps, n_alpha
    )
    alphas_upper = np.linspace(
        alpha_c + eps, high_bound, n_alpha
    )

    energies_lower = scan_energy(alphas_lower)
    energies_upper = scan_energy(alphas_upper)

    # Find local minima
    minima_lower = find_minima(alphas_lower, energies_lower)
    minima_upper = find_minima(alphas_upper, energies_upper)

    # Find global minimum
    candidates = minima_lower + minima_upper

    if not candidates:
        raise RuntimeError("No valid free energy minimum found.")

    alpha_opt, f_min = min(candidates, key=lambda x: x[1])

    # Retrieve optimal parameters
    f_min, beta_opt, a_opt, b_opt, c_opt = free_energy(
        alpha_opt, phi_w, phi_s, kappa, kappa_bar, H0
    )

    return alpha_opt, beta_opt, a_opt, b_opt, c_opt, f_min
