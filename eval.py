import numpy as np
import matplotlib.pyplot as plt
from helpers import diffraction_pattern

# assume input is 322x322 numpy arrays
# A is the "ground truth" AKA usually the multislice solution 
def deviation(result_psi, result_settings, ground_truth_psi, ground_truth_settings):
    result = diffraction_pattern(result_psi, result_settings)
    ground_truth = diffraction_pattern(ground_truth_psi, ground_truth_settings)

    if result.shape != ground_truth.shape:
        # error, cant compare
        return None

    # Compute the Frobenius norm of the difference
    deviation = np.linalg.norm(result - ground_truth, ord='fro') / np.linalg.norm(ground_truth, ord='fro')

    return deviation


# methods are being visualized AGAINST ground truth (another method)
def deviation_matrix(methods, ground_truth, voltages, run_result, alphas, dzs):
    # Initialize ax
    fig, ax = plt.subplots(len(voltages), len(methods), squeeze=False)

    for i in range(len(methods)):
        method = methods[i]
        for j in range(len(voltages)):
            voltage = voltages[j]

            # access method run in question
            psis = run_result[method][voltage][0]
            settings = run_result[method][voltage][1]

            # access corresponding ground truth
            psis_gt = run_result[ground_truth][voltage][0]
            settings_gt = run_result[ground_truth][voltage][1]

# methods are being visualized AGAINST ground truth (another method)
def deviation_matrix(methods, labels, ground_truth, voltages, run_result, alphas, dzs):
    # Make the subplot grid: rows = voltages, cols = methods; share axes across all plots
    fig, axes = plt.subplots(
        len(voltages), len(methods),
        squeeze=False, sharex=True, sharey=True,
        figsize=(4*len(methods), 3*len(voltages))
    )

    # Precompute x-limits: start around 20 if applicable, and invert (large -> small)
    dzs = list(dzs)
    dz_min, dz_max = min(dzs), max(dzs)
    x_left = max(20, dz_max)  # start
    x_right = dz_min          # end

    for i, method in enumerate(methods):
        for j, voltage in enumerate(voltages):
            ax = axes[j, i]

            # access method run in question
            psis = run_result[method][voltage][0]       # shape: [len(alphas)][len(dzs)]
            settings = run_result[method][voltage][1]   # same indexing or broadcastable

            # access corresponding ground truth
            psis_gt = run_result[ground_truth][voltage][0]
            settings_gt = run_result[ground_truth][voltage][1]

            # plot one curve per alpha
            for a_idx, alpha in enumerate(alphas):
                y_vals = []
                for d_idx, dz in enumerate(dzs):
                    res_psi = psis[a_idx][d_idx]
                    gt_psi = psis_gt[a_idx][d_idx]

                    res_settings = settings[a_idx][d_idx]
                    gt_settings = settings_gt[a_idx][d_idx]

                    dev = deviation(res_psi, res_settings, gt_psi, gt_settings)
                    y_vals.append(dev)

                ax.plot(dzs, y_vals, label=f"α={alpha}")

            # titles/labels
            ax.set_title(f"{labels[i]} @ {voltage} kV")
            ax.grid(True, alpha=0.3)

            # invert x-axis so decreasing dz goes to the right (shows convergence as dz ↓)
            ax.set_xlim(x_left, x_right)

    # Shared axis labels (set only on outer edges to avoid clutter)
    for col in range(len(methods)):
        axes[-1, col].set_xlabel("dz")
    for row in range(len(voltages)):
        axes[row, 0].set_ylabel("deviation")

    # One shared legend for all curves (uses first axes' handles)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    fig.tight_layout()
    plt.show()


def deviation_matrix(methods, labels, ground_truth, voltages, run_result, alphas, dzs):
    # rows = voltages, cols = methods; share axes across all plots
    fig, axes = plt.subplots(
        len(voltages), len(methods),
        squeeze=False, sharex=True, sharey=True,
        figsize=(4*len(methods), 3*len(voltages)),
        constrained_layout=True  # better spacing with an external legend
    )

    # Precompute x-limits: start around 20 if applicable, and invert (large -> small)
    dzs = list(dzs)
    dz_min, dz_max = min(dzs), max(dzs)
    x_range = max(1e-9, dz_max - dz_min)
    x_pad = 0.05 * x_range  # 5% padding
    x_left  = dz_max + x_pad   # start (left side, larger value)
    x_right = dz_min - x_pad            # end  (right side, smaller value)

    for i, method in enumerate(methods):
        for j, voltage in enumerate(voltages):
            ax = axes[j, i]

            # access method run in question
            psis = run_result[method][voltage][0]       # [len(alphas)][len(dzs)]
            settings = run_result[method][voltage][1]

            # access corresponding ground truth
            psis_gt = run_result[ground_truth][voltage][0]
            settings_gt = run_result[ground_truth][voltage][1]

            # plot one curve per alpha
            for a_idx, alpha in enumerate(alphas):
                y_vals = []
                for d_idx, dz in enumerate(dzs):
                    res_psi = psis[a_idx][d_idx]
                    gt_psi = psis_gt[a_idx][d_idx]
                    res_settings = settings[a_idx][d_idx]
                    gt_settings = settings_gt[a_idx][d_idx]
                    dev = deviation(res_psi, res_settings, gt_psi, gt_settings)
                    y_vals.append(dev)

                # add markers on each datapoint + slightly thicker line
                ax.plot(dzs, y_vals, marker='o', markersize=3, linewidth=1.5, label=f"α={alpha}")

            # titles/labels
            ax.set_title(f"{labels[i]} @ {voltage} kV")
            ax.grid(True, alpha=0.3)

            # invert x-axis so decreasing dz goes to the right, with padding
            ax.set_xlim(x_left, x_right)

            # add a little padding on y as well to avoid touching the frame
            ax.margins(x=0.03, y=0.10)

    # Shared axis labels (set only on outer edges to avoid clutter)
    for col in range(len(methods)):
        axes[-1, col].set_xlabel("dz")
    for row in range(len(voltages)):
        axes[row, 0].set_ylabel("deviation")

    # Build a single, visible legend outside the right edge, centered vertically
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, legend_labels,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=True, title="Curves"
        )

    # Leave a bit of room on the right for the legend (works with constrained_layout)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    plt.show()
