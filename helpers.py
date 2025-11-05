import numpy as np
from scipy import signal


def bin_z(f, dz, factor: int = 2):

    nz, ny, nx = f.shape
    nz = factor * (nz // factor)  # Enforce consistency

    f = np.sum(np.reshape(f[:nz, :, :], (factor, -1), order='F'), axis=0)

    return np.reshape(f, (-1, ny, nx), order='F'), dz * factor  # Don't forget to update dz


def crop_xy(f, factor: int = 2):

    _, ny, nx = f.shape

    dny = max(1, ny // (2 * factor))
    dnx = max(1, nx // (2 * factor))

    return f[:, (ny//2 - dny):(ny//2 + dny), (nx//2 - dnx):(nx//2 + dnx)]


def crop_z(f, factor: int = 2):

    nz, _, _ = f.shape

    dnz = max(1, nz // factor)

    return f[:dnz, :, :]


def amorphous_sample(nz=2000, ny=480, nx=None, seed=None, amplitude=15.):

    if nx is None:
        nx = ny + 0

    rand_gen = np.random.default_rng(seed=seed)

    return amplitude * rand_gen.random((nz, ny, nx))**5  # Dirty approx. of an amorphous sample


def fft2(f):

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f), norm='ortho'))


def ifft2(f):

    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f), norm='ortho'))


def bandwidth_limit(cfg):

    # The outer 3rd needs to be zero'd to prevent wrap-around artifacts
    n = cfg.shape[2]
    x = np.arange(0, n) - n//2
    rsq = x[:, None]**2 + x[None, :]**2

    return np.asarray(rsq < (.33 * n)**2, dtype=float)


def crop_dp(dp, cfg):
    # crop the area away that has been zero'd in the bandwidth limitation step

    n = cfg.shape[-1]
    dn = n//3 + 1

    return dp[(n//2 - dn):(n//2 + dn), (n//2 - dn):(n//2 + dn)]


def propagator(cfg):  # In Fourier space
    # usq = u**2 + v**2, with u and v coordinates in Fourier space

    n = cfg.shape[-1]
    u = cfg.du * (np.arange(0, n) - n//2)
    usq = u[:, None]**2 + u[None, :]**2

    return np.exp(-1.j * np.pi * cfg.lam * cfg.dz * usq)


def propagator_half(cfg):  # In Fourier space
    # usq = u**2 + v**2, with u and v coordinates in Fourier space

    n = cfg.shape[-1]
    u = cfg.du * (np.arange(0, n) - n//2)
    usq = u[:, None]**2 + u[None, :]**2

    return np.exp(-1.j * np.pi * cfg.lam * cfg.dz /2 * usq)


def diffraction_pattern(psi, cfg):

    # psi = multislice(potential, cfg)  # Calculate the exit wave of the sample

    dp = np.abs(fft2(psi))**2  # Convert to diffraction space and intensities

    return crop_dp(dp, cfg)


LAPLACIAN_KERNEL = 1 / 4 * np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]])
BOUNDARY_MODE = "symm"
CONVOLVE_MODE = "same"


def laplace_n(f, n):
    out = f.copy()  # might be inefficient, but assignment is required I think (Luc)
    for _ in range(n):
        out = signal.convolve2d(f, LAPLACIAN_KERNEL, boundary=BOUNDARY_MODE, mode=CONVOLVE_MODE)
    return out


def laplace(f, method = 1):
    if method == 1:
        # 5 point stencil
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif method == 2:
        # 9 point stencil for gamma = 1/2
        kernel = 1/4*np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]])
    elif method == 3:
        # 9 point stencil for gamma = 1/3
        kernel = 1/6*np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]])
    return signal.convolve2d(f, kernel, boundary="symm", mode="same")