from dotenv import load_dotenv
import os
import numpy as np

# Import scripts
from helpers import amorphous_sample
from simulation import run, multislice, fds
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
alphas = [alpha, alpha/1.2]
dzs = [dz, dz/2]

# get multislice results
psis_ms, settings_ms = run(multislice, potential, alphas, dzs)

# get fds results
# psis_fds = run(fds, potential, alphas, dzs)

# visualize result (multislice)
visualize_grid(psis_ms, alphas, dzs, settings_ms)

# compare results
#result = rel_error(ms_pattern, fds_pattern)