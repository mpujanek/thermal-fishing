import numpy as np
import time
import statistics
from datetime import timedelta
import pickle
from Settings import Settings
from helpers import bandwidth_limit, propagator, fft2, ifft2, crop_xy, crop_z, laplace, propagator_half, laplace_n, bin_z


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
    prop_half = propagator_half(cfg)

    # The multislice itself is surprisingly simple:
    psi = cfg.probe  # Initialize with the probe function
    
    psi = ifft2(fft2(psi) * prop_half * bwl_msk)
    
    for ii in range(cfg.shape[0]-1):
        tmp = fft2(np.exp(1.j * cfg.sigma * potential[ii, :, :] * cfg.dz) * psi)
        psi = ifft2(tmp * prop * bwl_msk)  # Impinging wave for the next slice
        
    tmp = fft2(np.exp(1.j * cfg.sigma * potential[-1, :, :] * cfg.dz) * psi)
    psi = ifft2(tmp * prop_half * bwl_msk)

    return psi


def fds(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)

    # The multislice itself is surprisingly simple:
    psi = np.copy(cfg.probe)  # Initialize with the probe function
    psi_prev = np.zeros(cfg.probe.shape)
    k = cfg.alpha/cfg.lam
    c_plus = 1+2*np.pi*1j*cfg.dz/cfg.lam
    c_minus = 1-2*np.pi*1j*cfg.dz/cfg.lam
    for ii in range(cfg.shape[0]):
        term1 = ifft2(-4 * (np.pi * k)**2 * fft2(psi))
        term2 = 4 * np.pi * cfg.sigma / cfg.lam * potential[ii, :, :] * psi
        tmp = 1 / c_plus * (2 * psi - cfg.dz**2 * (term1 + term2)) - c_minus / c_plus * psi_prev
        psi_next = np.copy(ifft2(fft2(tmp) * bwl_msk))

        psi_prev = np.copy(psi)
        psi = np.copy(psi_next)

    return psi


def fds_v2(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)
    prop = propagator(cfg)

    # Initial layer is given
    psi_prev = np.copy(cfg.probe)  # Initialize with the probe function
    
    # First layer computed through standard multislice
    tmp = fft2(np.exp(1.j * cfg.sigma * potential[0, :, :] * cfg.dz) * psi_prev)
    psi = ifft2(tmp * prop * bwl_msk)

    Nx, Ny = cfg.shape[1], cfg.shape[2]
    kx = np.fft.fftfreq(Nx, cfg.dx)  # spatial frequencies along x-axis
    ky = np.fft.fftfreq(Ny, cfg.dx)  # spatial frequencies along y-axis

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = KX**2 + KY**2

    c_plus = 1+2*np.pi*1j*cfg.dz/cfg.lam
    c_minus = 1-2*np.pi*1j*cfg.dz/cfg.lam
    for ii in range(1, cfg.shape[0]):
        term1 = ifft2(-4 * np.pi**2 * k * fft2(psi))
        term2 = 4 * np.pi * cfg.sigma / cfg.lam * potential[ii, :, :] * psi
        tmp = 1 / c_plus * (2 * psi - cfg.dz**2 * (term1 + term2)) - c_minus / c_plus * psi_prev
        psi_next = np.copy(ifft2(fft2(tmp)*bwl_msk))

        psi_prev = np.copy(psi)
        psi = np.copy(psi_next)

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
        term1 = laplace(psi, method =2)
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
    
    # start at ii=1 since we do the 0th iteration with multislice
    for ii in range(1, cfg.shape[0]):
        term1 = laplace(psi) / (cfg.dx**2)
        term2 = 4 * np.pi * cfg.sigma / cfg.lam * potential[ii, :, :] * psi
        tmp = 1 / c_plus * (2 * psi - cfg.dz**2 * (term1 + term2)) - c_minus / c_plus * psi_prev
        psi_next = np.copy(ifft2(fft2(tmp)*bwl_msk))

        psi_prev = np.copy(psi)
        psi = np.copy(psi_next)

    return psi


def fcms(potential, cfg):

    # Precompute the bandwidth limiting mask and the Fresnel propagator
    bwl_msk = bandwidth_limit(cfg)

    # The multislice itself is surprisingly simple:
    psi = cfg.probe  # Initialize with the probe function
    K0 = 1 / cfg.lam
    a = 2 * np.pi * 1.j * cfg.dz * K0
    c = 1 / (2 * np.pi * K0)**2
    for ii in range(cfg.shape[0]):
        b = 1 + cfg.sigma / (np.pi * K0) * potential[ii, :, :]
        coef0 = np.exp(a * (np.sqrt(b) - 1))
        coef1 = a * c * np.exp(a * (np.sqrt(b) - 1)) / (2 * np.sqrt(b))
        coef2 = a * (a * np.sqrt(b) - 1) * c**2 * np.exp(a * (np.sqrt(b) - 1)) / (8 * np.pow(b, 3/2))
        coef3 = a * (3 - 3 * a * np.sqrt(b) + a**2 * b) * c**3 * np.exp(a * (np.sqrt(b) - 1)) / (48 * np.pow(b, 5/2))
        term0 = coef0 * psi
        term1 = coef1 * laplace(psi)
        term2 = coef2 * laplace_n(psi, 2)
        term3 = coef3 * laplace_n(psi, 3)
        tmp = term0 + term1 + term2 + term3
        psi = ifft2(fft2(tmp) * bwl_msk)  # Impinging wave for the next slice

    return psi


# x,y factor=1; z cropping up to us (can use factor 5 or 10); vary dz
def run(solver, potential, voltage, alphas, dzs):
    # Optional: select inner quarter to compute faster during testing by cropping the x- and y-directions
    potential = crop_xy(potential, factor=1)
    # Crop the z-direction because sample too thick
    potential = crop_z(potential, factor=10)

    # default values [pm]
    dx = 20.6
    dz = 20.6

    psis = []
    settings = []

    total = len(alphas) * len(dzs)
    times = []
    t0_all = time.perf_counter()

    current = 0
    for i in range(len(alphas)):
        psis.append([])
        settings.append([])
        for j in range(len(dzs)):
            current += 1
            t0_iter = time.perf_counter()

            print(f"\n[{current}/{total}] Running {solver.__name__} on alpha={alphas[i]}, dz={dzs[j]*20.6}")

            # Bin the z-direction:
            potential, dz = bin_z(potential, dz, factor=dzs[j])

            cfg = Settings(ht=voltage,  # [kV] 'high tension,' a.k.a. acceleration voltage.  Vary between 10. and 100.
                    # The size and dx of the provided potential are optimized for alpha=20. Keep fixed, especially
                    # in the beginning of the assignment! Later you can vary between 10. and 30. if you're curious.
                    alpha=alphas[i],  # [mrad] convergence angle, 20. is the default.
                    shape=potential.shape,  # shape of the potential array: z-, y- and x-direction
                    dx=dx,  # [pm] sampling size in y and x direction
                    dz=dz,  # [pm] sampling size in z direction, same as dx when None
                    )
            
            psi = solver(potential, cfg)

            # Timing stats
            dt = time.perf_counter() - t0_iter
            times.append(dt)
            avg = statistics.mean(times)
            remaining = total - current
            eta = timedelta(seconds=int(remaining * avg))

            print(f"  Finished in {dt:.2f}s | avg {avg:.2f}s/it | block ETA {eta}")

            psis[i].append(psi)
            settings[i].append(cfg)

    total_time = time.perf_counter() - t0_all
    print(f"\nAll {total} iterations finished in {timedelta(seconds=int(total_time))} "
          f"(avg {statistics.mean(times):.2f}s/iteration)")

    return psis, settings


def big_run(methods, voltages, potential, alphas, dzs, save_path="results.pkl"):
    result = {}
    times = []

    total = len(methods) * len(voltages)
    current = 0
    t0_all = time.perf_counter()

    for method in methods:
        result[method.__name__] = {}  # Use method name (not function object) for serialization
        for voltage in voltages:
            current += 1
            t0_iter = time.perf_counter()
            print(f"\n=== ({current}/{total}) Running full block: {method.__name__} @ {voltage} kV ===")

            psis, settings = run(method, potential, voltage, alphas, dzs)

            # Timing stats
            dt = time.perf_counter() - t0_iter
            times.append(dt)
            avg = statistics.mean(times)
            remaining = total - current
            eta = timedelta(seconds=int(remaining * avg))
            print(f"Block finished in {dt:.2f}s | avg/block {avg:.2f}s | total ETA {eta}")

            result[method.__name__][voltage] = (psis, settings)

            # Save intermediate progress every iteration
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            print(f"Progress saved to {save_path}")

    total_time = time.perf_counter() - t0_all
    print(f"\nAll {total} blocks finished in {timedelta(seconds=int(total_time))} "
          f"(avg {statistics.mean(times):.2f}s/block)")
    print(f"Final results saved to {save_path}")

    return result
