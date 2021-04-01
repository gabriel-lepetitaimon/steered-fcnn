import numpy as np


def max_steerable_harmonics(radius):
    if radius == 0:
        return 0

    def circle_area(r):
        return np.pi * r ** 2

    inter_area = circle_area(radius + .5) - circle_area(radius - .5)
    return int(inter_area//2)-1


def radial_steerable_filter(size, k, r, std=.5):
    from ..utils.rotequivariance_toolbox import polar_space
    rho, phi = polar_space(size)
    G = np.exp(-(rho-r)**2/(2 * std**2)) / (std * np.sqrt(2*np.pi))
    if k != 0:
        G[rho == 0] *= 0
    PHI = np.exp(1j*k*phi)

    f = G*PHI
    return f


def plot_filter(F, axis=True, spd=False):
    from ..utils.rotequivariance_toolbox import polar_spectral_power
    import torch
    import matplotlib.pyplot as plt
    if isinstance(F, torch.Tensor):
        F = F.detach().cpu().numpy()
    h, w = F.shape
    v = max(F.max(), -F.min())

    if spd is True:
        spd = 16

    if spd:
        fig, (ax_filt, ax_spd) = plt.subplots(1, 2)
    else:
        fig, ax_filt = plt.subplots()
        ax_spd = None

    # --- PLOT FILTER ---
    ax_filt.imshow(-F, interpolation='none', vmin=-v, vmax=v, aspect='equal', cmap='RdGy')
    if axis:
        # Major ticks
        ax_filt.set_xticks(np.arange(0, w, 1))
        ax_filt.set_yticks(np.arange(0, h, 1))

        # Labels for major ticks
        ax_filt.set_xticklabels(np.arange(1, w+1, 1))
        ax_filt.set_yticklabels(np.arange(1, h+1, 1))
    else:
        ax_filt.set_xticklabels([])
        ax_filt.set_yticklabels([])
        ax_filt.set_xticks([])
        ax_filt.set_yticks([])

    # Minor ticks
    ax_filt.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax_filt.set_yticks(np.arange(-.5, h, 1), minor=True)

    # Gridlines based on minor ticks
    ax_filt.grid(which='minor', color='w', linestyle='-', linewidth=2)

    # --- PLOT SPD ---
    if spd:
        polar_spectral_power(F, plot=ax_spd, theta=spd)
    fig.tight_layout(w_pad=-3)
    fig.show()
