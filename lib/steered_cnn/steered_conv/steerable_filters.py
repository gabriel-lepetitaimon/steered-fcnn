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
        G *= np.sqrt(2)
    PHI = np.exp(1j*k*phi)

    f = G*PHI
    return f


def plot_filter(F, axis=True, spd=False, plot=None, colorbar=False):
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
    elif plot is not None:
        ax_filt = plot
        fig = None
    else:
        fig, ax_filt = plt.subplots()
        ax_spd = None

    # --- PLOT FILTER ---
    im = ax_filt.imshow(-F, interpolation='none', vmin=-v, vmax=v, aspect='equal', cmap='RdGy')
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

    if colorbar:
        plt.colorbar(im)

    # --- PLOT SPD ---
    if spd:
        polar_spectral_power(F, plot=ax_spd, theta=spd)
    if fig:
        fig.tight_layout(w_pad=-3)
        fig.show()


def cos_sin_ka(cos_sin_a, cos_sin_km1_a):
    """
    Computes cos((k+1)α) and sin((k+1)α) given cos(α), sin(α), cos(kα) and sin(kα):
         cos((k+1)α) = cos(kα)cos(α) - sin(kα)sin(α)
         sin((k+1)α) = cos(kα)sin(α) + sin(kα)cos(α)
    Args:
        cos_sin_a: A tensor of shape [2, n1, n2, ...], where cos1_sin1[0]=cos(α) and cos1_sin1[1]=sin(α).
        cos_sin_km1_a: A tensor of shape [2, n1, n2, ...], where coskm1_sinkm1[0]=cos(kα) and coskm1_sinkm1[1]=sin(kα).

    Returns: A tensor cosk_sink of shape [2, n1, n2, ...], where cosk_sink[0]=cos((k+1)α) and cosk_sink[1]=sin((k+1)α).
    """
    import torch
    return torch.stack((
        cos_sin_km1_a[0] * cos_sin_a[0] - cos_sin_km1_a[1] * cos_sin_a[1],
        cos_sin_km1_a[0] * cos_sin_a[1] + cos_sin_km1_a[1] * cos_sin_a[0]
    ))


def cos_sin_ka_stack(cos_alpha, sin_alpha, k):
    """
    Computes the matrix:
    [[cos(α), cos(2α), ..., cos(kα)],
     [sin(α), sin(2α), ..., sin(kα)]]

    Args:
        cos_alpha: The tensor cos(α) of shape [n0, n1, ...]
        sin_alpha: The tensor sin(α) of shape [n0, n1, ...]
        k: The max k.

    Returns: The tensor cos_sin_ka of shape [2, k, n0, n1, ...], where:
                cos_sin_ka[0] = [cos(α), cos(2α), ..., cos(kα)]
                cos_sin_ka[1] = [sin(α), sin(2α), ..., sin(kα)]
    """
    import torch
    cos_sin_alpha = torch.stack([cos_alpha, sin_alpha])
    cos_sin_km1_alpha = cos_sin_alpha
    r = [cos_sin_alpha]
    for i in range(2, k+1):
        cos_sin_km1_alpha = cos_sin_ka(cos_sin_alpha, cos_sin_km1_alpha)
        r += [cos_sin_km1_alpha]
    return torch.stack(r, dim=1)
