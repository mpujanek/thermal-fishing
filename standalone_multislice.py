# Author: Wouter Van den Broek [1]
# Data cube with potential: Holger Kohr [1]
# [1] Thermo Fisher Scientific, Eindhoven, The Netherlands, October 2025.
# License: CC BY-NC-ND, https://creativecommons.org/licenses/by-nc-nd/4.0/

# backend for graphic generation
import numpy as np
import matplotlib
from helpers import laplace, laplace_v2
matplotlib.use("Qt5Agg")

# use path from .env
from dotenv import load_dotenv
import os
load_dotenv()
path = os.getenv("POT_PATH")

class Settings:

    def __init__(self,
                 ht: float = 100.,  # [kV] 'high tension,' a.k.a. acceleration voltage
                 alpha: float = 20.,  # [mrad] convergence angle
                 shape: tuple = (2000, 480, 480),  # shape of the potential array: z, y and x direction
                 dx: float = 20.6,  # [pm] sample size in y and x direction
                 dz: float = None,  # [pm] sample size in z direction
                 ):

        self.ht = ht
        self.alpha = alpha
        self.shape = shape
        if self.shape[1] != self.shape[2]:
            print('x- and y-dimensions must be equal!')
            self.shape = None  # Make it crash
        self.dx = dx
        if dz is None:
            self.dz = dx
        else:
            self.dz = dz
        self.du = None  # Pixel size in reciprocal space

        self.htr = None  # Relativistic acceleration voltage
        self.lam = None  # Electron wavelength
        self.width = None  # Lateral width of the beam
        self.dof = None  # Depth of focus, i.e. vertical beam width
        self.sigma = None  # Interaction parameter
        self.aperture = None  # Beam-forming aperture in reciprocal space
        self.probe = None  # Wavefunction of the probe

        self.initialize()

    def initialize(self):
        self.alpha *= 1e-3  # [rad] Transform from mrad to rad
        self.ht *= 1e3  # [V] Transform high tension to Volts
        self.dx *= 1e-3  # Transform to nm
        self.dz *= 1e-3  # Transform to nm
        self.du = 1. / (self.shape[-1] * self.dx)  # [1/nm]
        self.set_relativistic_voltage()  # Set the relativistic voltage htr
        self.set_wavelength()  # [nm] Wavelength of the electron
        self.set_beam_width()  # [nm] Width of the beam, spatial resolution.
        self.set_depth_of_field()  # [nm]
        self.set_interaction_parameter()  # [1/nm/V] Interaction parameter
        self.set_aperture()  # Precompute the aperture
        self.set_probe()  # Precompute the probe

    def set_relativistic_voltage(self):
        self.htr = self.ht * (1. + self.ht * 0.9784755918e-6)

    def set_wavelength(self):
        self.lam = 1.22642596588 / np.sqrt(self.htr)

    def set_beam_width(self):
        self.width = .50 * self.lam / self.alpha  # Diffraction limit

    def set_depth_of_field(self):
        self.dof = 2. * self.lam / self.alpha ** 2

    def set_interaction_parameter(self):
        mec2 = 0.5109989500e6  # [eV]
        self.sigma = 2. * np.pi * ((mec2 + self.ht) / (2. * mec2 + self.ht)) / (self.lam * self.ht)

    def set_aperture(self):
        n = self.shape[-1]
        u = self.du * (np.arange(0, n) - n//2)
        usq = u[:, None]**2 + u[None, :]**2
        self.aperture = np.zeros_like(usq)
        self.aperture[usq < (self.alpha / self.lam)**2] = 1.

    def set_probe(self):
        self.probe = ifft2(self.aperture / np.sum(self.aperture**2))


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


def propagator(cfg):  # In Fourier space
    # usq = u**2 + v**2, with u and v coordinates in Fourier space

    n = cfg.shape[-1]
    u = cfg.du * (np.arange(0, n) - n//2)
    usq = u[:, None]**2 + u[None, :]**2

    return np.exp(-1.j * np.pi * cfg.lam * cfg.dz * usq)


def multislice(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)
    prop = propagator(cfg)

    # The multislice itself is surprisingly simple:
    psi = cfg.probe  # Initialize with the probe function
    for ii in range(cfg.shape[0]):
        tmp = fft2(np.exp(1.j * cfg.sigma * potential[ii, :, :] * cfg.dz) * psi)
        psi = ifft2(tmp * prop * bwl_msk)  # Impinging wave for the next slice

    return psi


def multislice_alt(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)
    prop = propagator(cfg)

    # The multislice itself is surprisingly simple:
    psi = cfg.probe  # Initialize with the probe function
    for ii in range(cfg.shape[0]):
        tmp = np.exp(1.j * cfg.sigma * potential[ii, :, :] * cfg.dz)*ifft2(fft2(psi)*prop*bwl_msk)
        psi = tmp  # Impinging wave for the next slice

    return psi


def multislice_v2(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)
    prop = propagator(cfg)
    prop_half = prop/2

    # The multislice itself is surprisingly simple:
    psi = cfg.probe  # Initialize with the probe function
    
    psi = ifft2(fft2(psi) * prop_half * bwl_msk)
    
    for ii in range(cfg.shape[0]-1):
        tmp = fft2(np.exp(1.j * cfg.sigma * potential[ii, :, :] * cfg.dz) * psi)
        psi = ifft2(tmp * prop * bwl_msk)  # Impinging wave for the next slice
        
    tmp = fft2(np.exp(1.j * cfg.sigma * potential[-1, :, :] * cfg.dz) * psi)
    psi = ifft2(tmp * prop_half * bwl_msk)

    return psi


def fds_conv(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)

    # The multislice itself is surprisingly simple:
    psi = np.copy(cfg.probe)  # Initialize with the probe function
    psi_prev = np.zeros_like(cfg.probe)   # Initialize with zeros

    c_plus = 1+2*np.pi*1j*cfg.dz/cfg.lam
    c_minus = 1-2*np.pi*1j*cfg.dz/cfg.lam
    for ii in range(cfg.shape[0]):
        term1 = laplace(psi)
        term2 = 4 * np.pi * cfg.sigma / cfg.lam * potential[ii, :, :] * psi
        tmp = 1 / c_plus * (2 * psi - cfg.dz**2 * (term1 + term2)) - c_minus / c_plus * psi_prev
        psi_next = np.copy(ifft2(fft2(tmp)*bwl_msk))

        psi_prev = np.copy(psi)
        psi = np.copy(psi_next)

    return psi


def fds_conv_v2(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)
    prop = propagator(cfg)

    # Initial layer is given
    psi_prev = np.copy(cfg.probe)  # Initialize with the probe function
    
    # First layer computed through standard multislice
    tmp = fft2(np.exp(1.j * cfg.sigma * potential[0, :, :] * cfg.dz) * psi_prev)
    psi = ifft2(tmp * prop * bwl_msk)
    
    c_plus = 1+2*np.pi*1j*cfg.dz/cfg.lam
    c_minus = 1-2*np.pi*1j*cfg.dz/cfg.lam
    for ii in range(cfg.shape[0]):
        term1 = laplace_v2(psi) * (cfg.dx**2)
        term2 = 4 * np.pi * cfg.sigma / cfg.lam * potential[ii, :, :] * psi
        tmp = 1 / c_plus * (2 * psi - cfg.dz**2 * (term1 + term2)) - c_minus / c_plus * psi_prev
        psi_next = np.copy(ifft2(fft2(tmp)*bwl_msk))

        psi_prev = np.copy(psi)
        psi = np.copy(psi_next)

    return psi


def fds(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)

    # The multislice itself is surprisingly simple:
    psi = np.copy(cfg.probe)  # Initialize with the probe function
    psi_prev = np.zeros_like(cfg.probe)   # Initialize with zeros

    Nx, Ny = cfg.shape[1], cfg.shape[2]
    kx = np.fft.fftfreq(Nx, cfg.dx)  # spatial frequencies along x-axis
    ky = np.fft.fftfreq(Ny, cfg.dx)  # spatial frequencies along y-axis

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = KX**2 + KY**2

    c_plus = 1+2*np.pi*1j*cfg.dz/cfg.lam
    c_minus = 1-2*np.pi*1j*cfg.dz/cfg.lam
    for ii in range(cfg.shape[0]):
        term1 = ifft2(-4 * np.pi**2 * k * fft2(psi))
        term2 = 4 * np.pi * cfg.sigma / cfg.lam * potential[ii, :, :] * psi
        tmp = 1 / c_plus * (2 * psi - cfg.dz**2 * (term1 + term2)) - c_minus / c_plus * psi_prev
        psi_next = np.copy(ifft2(fft2(tmp)*bwl_msk))

        psi_prev = np.copy(psi)
        psi = np.copy(psi_next)

    return psi


def crop_dp(dp, cfg):
    # crop the area away that has been zero'd in the bandwidth limitation step

    n = cfg.shape[-1]
    dn = n//3 + 1

    return dp[(n//2 - dn):(n//2 + dn), (n//2 - dn):(n//2 + dn)]


def diffraction_pattern(potential, cfg):

    psi = fds_conv(potential, cfg)  # Calculate the exit wave of the sample

    #psi = multislice_v2(potential, cfg)

    dp = np.abs(fft2(psi))**2  # Convert to diffraction space and intensities

    return crop_dp(dp, cfg)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dx = 20.6  # [pm]
    dz = dx + 0.  # same as dx

    # Read in your potential here:
    potential = np.load(path)

    # Optional: another array for testing:
    # potential = amorphous_sample(seed=31415)  # Set the seed so the potential is the same between runs

    # Bin the z-direction to test various dz samplings:
    # potential, dz = bin_z(potential, dz, factor=10)
    # Optional: select inner quarter to compute faster during testing by cropping the x- and y-directions
    potential = crop_xy(potential, factor=1)
    # Crop the z-direction if needed
    potential = crop_z(potential, factor=20)

    settings = Settings(ht=30.,  # [kV] 'high tension,' a.k.a. acceleration voltage.  Vary between 10. and 100.
                        # The size and dx of the provided potential are optimized for alpha=20. Keep fixed, especially
                        # in the beginning of the assignment! Later you can vary between 10. and 30. if you're curious.
                        alpha=20.,  # [mrad] convergence angle, 20. is the default.
                        shape=potential.shape,  # shape of the potential array: z-, y- and x-direction
                        dx=dx,  # [pm] sampling size in y and x direction
                        dz=dz,  # [pm] sampling size in z direction, same as dx when None
                        )

    print(settings.dx, settings.dz, potential.shape, settings.sigma)

    # Compute the pattern
    pattern = diffraction_pattern(potential, settings)

    print(np.array(pattern).shape)

    # Show the results
    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.abs(settings.probe)**2, cmap='gray', interpolation='nearest')
    ax[0, 0].set_title('Electron probe')
    ax[0, 1].imshow(np.sum(potential, axis=0) * settings.sigma * settings.dz, cmap='gray', interpolation='nearest')
    ax[0, 1].set_title('Projected potential in radians')
    ax[1, 0].imshow(pattern, cmap='gray', interpolation='nearest')
    ax[1, 0].set_title('Diffraction pattern (linear gray scale)')
    ax[1, 1].imshow(np.log(1e9 * np.abs(pattern) + 1.), cmap='gray', interpolation='nearest')
    ax[1, 1].set_title('Diffraction pattern (logarithmic gray scale)')
    plt.show()
