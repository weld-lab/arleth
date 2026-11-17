from copy import deepcopy

from minimization import minimize_free_energy
from parameters import Parameters

"""
D[f,2](x) = (f(x+∂) - 2f(x) + f(x-∂))/∂^2
"""

""" TODO
    There might exists actually a better solution than computing 9 minimizations
    lying in Schur complement. I need to expand a little on this later.
"""

def compute_hessian(parameters, delta=1e-3):

    s0 = parameters.phi_s
    w0 = parameters.phi_w

    def g(ds, dw):
        p = parameters._replace(
            phi_s=s0 + ds,
            phi_w=w0 + dw
        )

        _,_,_,_,_, f = minimize_free_energy(p)

        return f

    g0 = g(0, 0)

    # second derivative w.r.t. phi_s
    A = (g(delta, 0) - 2*g0 + g(-delta, 0)) / delta**2

    # second derivative w.r.t. phi_s
    D = (g(0, delta) - 2*g0 + g(0, -delta)) / delta**2

    # cross terms
    B = (
        g(delta, delta)
        - g(delta, -delta)
        - g(-delta, delta)
        + g(-delta, -delta)
    ) / (4*delta**2)

    return A*D - B**2


