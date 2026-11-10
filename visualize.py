import numpy as np
import matplotlib.pyplot as plt

from fixed_params import kc


def visualize_microemulsion(alpha, beta, a, b, c,
                            N=128, dx=1.0, seed=None):

    """Generate a 3D Gaussian random field and display a 2D slice."""

    rng = np.random.default_rng(seed)
    L = N * dx

    # generate gaussian white noise
    noise = rng.standard_normal((N, N, N))

    # compute Fourier-space wavevectors (nm^-1)
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=dx)

    k2 = (
        kx[:, None, None]**2
        + ky[None, :, None]**2
        + kz[None, None, :]**2
    )

    # spectral density (eqn 15)
    spectrum = np.zeros_like(k2)

    mask = (k2 > 0) & (k2 <= kc**2)
    denominator = k2**2 - b*k2 + c

    if np.any(denominator[mask] <= 0):
        raise ValueError("The spectral density is not positive.")

    spectrum[mask] = a / denominator[mask]

    noise_k = np.fft.rfftn(noise)
    field_k = noise_k * np.sqrt(spectrum / dx**3)

    field = np.fft.irfftn(field_k, s=(N, N, N))


    if alpha > beta:
        water = (field >= alpha) | (field <= beta)
    else:
        water = (field >= alpha) & (field <= beta)


    z = N // 2
    slice_2d = field[:, :, z].T

    x = np.arange(N) * dx

    fig, ax = plt.subplots(figsize=(7, 7))
 
    yellow = "#F3B61F" ; blue =  "#255C99" ; red = "#D62828" 
    if alpha > beta:
        # Water-in-oil
        levels = [-np.inf, beta, alpha, np.inf]
        colors = [blue, yellow, blue]
    else:
        # Oil-in-water
        levels = [-np.inf, alpha, beta, np.inf]
        colors = [yellow, blue, yellow]

    ax.contourf(
        x, x, slice_2d,
        levels=levels,
        colors=colors,
        antialiased=True
    )

    ax.contour(
        x, x, slice_2d,
        levels=[min(alpha, beta), max(alpha, beta)],
        colors=red,
        linewidths=2.0,
        linestyles="solid"
    )

    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$y$ (nm)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    

    return field, water
