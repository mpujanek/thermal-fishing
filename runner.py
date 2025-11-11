from dotenv import load_dotenv
import os
import numpy as np

# Import scripts
from helpers import amorphous_sample
from simulation import run, big_run, multislice, fds, fds_conv, fds_v2, fds_conv_v2, multislice_v2, fcms
from visualization import visualize_grid
from eval import deviation_matrix, deviation_matrix_by_alpha, deviation_matrix_by_alpha_transpose, deviation_row_by_method_vs_voltage

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

alphas = [10., 20., 30.] # [mrad] convergence angle, 20. is the default.
dzs = [4, 3, 2, 1] # factors for dz increase + binning (positive integers)

methods = [fds_conv_v2, fcms]
labels = ["FDS", "FCMS"]
ground_truth = multislice
voltages = [100., 50., 30., 10.] # kV, 100. is the default

# get results
#psis_fcms, settings_fcms = run(fcms, potential, 100., alphas, dzs)
#psis_fcms50, settings_fcms50 = run(fcms, potential, 50., alphas, dzs)

#psis_fdsconv, settings_fdsconv = run(fds_conv_v2, potential, 100., alphas, dzs)
#psis_fdsconv50, settings_fdsconv50 = run(fds_conv_v2, potential, 50., alphas, dzs)

#psis_ms, settings_ms = run(multislice, potential, 100., alphas, dzs)
#psis_ms50, settings_ms50 = run(multislice, potential, 50., alphas, dzs)

save_path = "the_run.pkl"
#results = big_run(methods + [ground_truth], voltages, potential, alphas, dzs, save_path=save_path)

# visualize results
#visualize_grid(psis_fcms, alphas, dzs, settings_fcms, label="FCMS (100kV)")
#visualize_grid(psis_fcms50, alphas, dzs, settings_fcms50, label="FCMS (50kV)")

#visualize_grid(psis_fdsconv, alphas, dzs, settings_fdsconv, label="FDS Conv v2 (100kV)")
#visualize_grid(psis_fdsconv50, alphas, dzs, settings_fdsconv50, label="FDS Conv v2 (50kV)")

#visualize_grid(psis_ms, alphas, dzs, settings_ms, label="Multislice (100kV)")
#visualize_grid(psis_ms50, alphas, dzs, settings_ms50, label="Multislice (50kV)")

deviation_matrix_by_alpha_transpose(methods, labels, ground_truth, voltages, save_path, alphas, dzs, "the_deviation_transpose_NOTINVERT.png")

# compare results
#result = rel_error(ms_pattern, fds_pattern)