from dotenv import load_dotenv
import os
import numpy as np

# Import scripts
from helpers import amorphous_sample
from simulation import run, big_run, multislice, fds, fds_conv, fds_v2, fds_conv_v2, multislice_v2
from visualization import visualize_grid
from eval import deviation_matrix

# Load the .env file
load_dotenv()
# Access path
path = os.getenv("POT_PATH")
# Load dataset
potential = np.load(path)
# Optional: another array for testing:
# potential = amorphous_sample(seed=31415)  # Set the seed so the potential is the same between runs


# Set params for full run
dz = 20.6 # [pm], default

alphas = [10.] # [mrad] convergence angle, 20. is the default.
dzs = [dz/8, dz/16, dz/32, dz/64] #, dz/4, dz/8]

methods = [fds_conv_v2]
labels = ["FDS Conv v2"]
ground_truth = multislice
voltages = [100.] # kV, 100. is the default

# get results
#psis_ms, settings_ms = run(multislice, potential, alphas, dzs)
results = big_run(methods + [ground_truth], voltages, potential, alphas, dzs)

# visualize results
#visualize_grid(psis_ms, alphas, dzs, settings_ms, label="Multislice (50kV)")
deviation_matrix(methods, labels, ground_truth, voltages, results, alphas, dzs)

# compare results
#result = rel_error(ms_pattern, fds_pattern)