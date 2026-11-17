from typing import NamedTuple

class Parameters(NamedTuple):
    phi_w: float # polar/water volume fraction
    phi_s: float # surfactant volume fraction
    kappa: float # bending rigidity in kT units
    kappa_bar: float # gaussian bending rigidity in kT units
    H0: float # spontaneous mean curvature in nm^-1
    kc: float # spectral wavenumber cutoff in nm^-1 -> might be moved to specific place later
    ds: float # surfactant film thickness in nm
