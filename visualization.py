import matplotlib.pyplot as plt
import numpy as np
from helpers import diffraction_pattern

def visualize(psi, potential, settings):
    # Compute the pattern
    pattern = diffraction_pattern(psi, settings)

    #print(settings.dx, settings.dz, potential.shape, settings.sigma)
    #print(np.array(pattern).shape)

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

def visualize_grid(psis, alphas, dzs, settings, label=None):
    # Initialize ax
    fig, ax = plt.subplots(len(alphas), len(dzs), squeeze=False)

    if label is not None:
        fig.suptitle(f'Diffraction pattern (logarithmic gray scale); method: {label}')
    else:
        fig.suptitle('Diffraction pattern (logarithmic gray scale)')

    for i in range(len(alphas)):
        for j in range(len(dzs)):
            pattern = diffraction_pattern(psis[i][j], settings[i][j])

            ax[i, j].imshow(np.log(1e9 * np.abs(pattern) + 1.), cmap='gray', interpolation='nearest')
            ax[i, j].set_title(f'alpha = {alphas[i]:.2f}mrad, dz = {dzs[j]:.2f}pm')

    plt.show()