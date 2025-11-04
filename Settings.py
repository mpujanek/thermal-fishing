from helpers import ifft2
import numpy as np

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