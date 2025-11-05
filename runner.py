from dotenv import load_dotenv
import os
import numpy as np

# Import scripts
from helpers import amorphous_sample
from simulation import run, multislice, fds, fds_conv
from visualization import visualize_grid
from eval import rel_error

# Load the .env file
load_dotenv()
# Access path
path = os.getenv("POT_PATH")
# Load dataset
potential = np.load(path)
# Optional: another array for testing:
# potential = amorphous_sample(seed=31415)  # Set the seed so the potential is the same between runs


# Set params
alpha = 20.  # [mrad] convergence angle, 20. is the default.
dz = 20.6 # [pm], default

# Inputs for full run
alphas = [alpha/2, alpha, 2*alpha, 4*alpha]
dzs = [dz, dz/2, dz/4, dz/8]

# get results
psis_ms, settings_ms = run(multislice, potential, alphas, dzs)
psis_fds, settings_fds = run(fds, potential, alphas, dzs)
psis_fds_conv, settings_fds_conv = run(fds_conv, potential, alphas, dzs)

# visualize results
visualize_grid(psis_ms, alphas, dzs, settings_ms, label="Multislice")
visualize_grid(psis_fds, alphas, dzs, settings_fds, label="FDS")
visualize_grid(psis_fds_conv, alphas, dzs, settings_fds_conv, label="FDS (conv)")

# compare results
#result = rel_error(ms_pattern, fds_pattern)