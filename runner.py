from dotenv import load_dotenv
import os
import numpy as np

# Import scripts
from simulation import run, multislice, fds
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
alphas = [alpha]
dzs = [dz]

# get multislice results
psis_ms = run(multislice, potential, alphas, dzs)

# get fds results
psis_fds = run(fds, potential, alphas, dzs)

# compare result
for ms in psis_ms:
    for fds in psis_fds:
        ms_pattern = visualize(ms)
        fds_pattern = visualize(fds)
        result = rel_error(ms_pattern, fds_pattern)