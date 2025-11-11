import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
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
            ax.set_title(f"{labels[i]} at {voltage} kV")
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


"""
Plot a matrix of deviation curves with one subplot per (method, voltage) pair.
Each subplot shows different alphas as curves over dz.

Rows: voltages
Columns: methods
Curves: alphas
"""
def deviation_matrix(methods, labels, ground_truth, voltages, run_result, alphas, dzs):
    # Allow run_result to be either a dict or a path to a saved file
    if isinstance(run_result, (str, Path)):
        path = Path(run_result)
        print(f"Loading run results from {path} ...")
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                run_result = pickle.load(f)
        elif path.suffix == ".json":
            import json
            with open(path, "r") as f:
                run_result = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        print("Loaded successfully.")

    # map methods to method names
    methods = [method.__name__ for method in methods]
    ground_truth = ground_truth.__name__

    # rows = voltages, cols = methods; share axes across all plots
    fig, axes = plt.subplots(
        len(voltages), len(methods),
        squeeze=False, sharex=False, sharey=False,
        figsize=(4*len(methods), 3*len(voltages)),
        constrained_layout=True  # better spacing with an external legend
    )

    # Precompute x-limits: start around 20 if applicable, and invert (large -> small)
    dzs = [20.6 * dz for dz in dzs]
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
            ax.set_title(f"{labels[i]} at {voltage} kV")
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



"""
Plot a matrix of deviation curves with one subplot per (alpha, voltage) pair.
Each subplot shows different methods as curves over dz.

Rows: voltages
Columns: alphas
Curves: methods
"""
def deviation_matrix_by_alpha(methods, labels, ground_truth, voltages, run_result, alphas, dzs, filename):

    # Allow run_result to be either a dict or a path to a saved file
    if isinstance(run_result, (str, Path)):
        path = Path(run_result)
        print(f"Loading run results from {path} ...")
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                run_result = pickle.load(f)
        elif path.suffix == ".json":
            import json
            with open(path, "r") as f:
                run_result = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        print("Loaded successfully.")

    # Map to method names
    methods = [method.__name__ for method in methods]
    ground_truth = ground_truth.__name__

    # rows = voltages, cols = alphas; share axes
    fig, axes = plt.subplots(
        len(voltages), len(alphas),
        squeeze=False, sharex=False, sharey=True,
        figsize=(4 * len(alphas), 3 * len(voltages)),
        constrained_layout=True
    )

    # Compute scaled dzs and x-limits
    dzs = [20.6 * dz for dz in dzs]
    dz_min, dz_max = min(dzs), max(dzs)
    x_range = max(1e-9, dz_max - dz_min)
    x_pad = 0.05 * x_range
    x_left = dz_max + x_pad
    x_right = dz_min - x_pad

    for a_idx, alpha in enumerate(alphas):
        for j, voltage in enumerate(voltages):
            ax = axes[j, a_idx]

            # access ground truth for this voltage
            psis_gt = run_result[ground_truth][voltage][0]
            settings_gt = run_result[ground_truth][voltage][1]

            for m_idx, method in enumerate(methods):
                psis = run_result[method][voltage][0]
                settings = run_result[method][voltage][1]

                # Compute deviations for each dz at this alpha
                y_vals = []
                for d_idx, dz in enumerate(dzs):
                    res_psi = psis[a_idx][d_idx]
                    gt_psi = psis_gt[a_idx][d_idx]
                    res_settings = settings[a_idx][d_idx]
                    gt_settings = settings_gt[a_idx][d_idx]
                    dev = deviation(res_psi, res_settings, gt_psi, gt_settings)
                    y_vals.append(dev)

                ax.plot(dzs, y_vals, marker='o', markersize=3, linewidth=1.5, label=labels[m_idx])

            # Titles/labels
            ax.set_title(f"α={alpha} at {voltage} kV")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(-1, 10)
            #ax.set_yscale('log')
            ax.margins(x=0.03, y=0.10)

    # Shared axis labels (outer edges only)
    for col in range(len(alphas)):
        axes[-1, col].set_xlabel("dz")
    for row in range(len(voltages)):
        axes[row, 0].set_ylabel("Deviation from multislice")

    # Build a single shared legend using method names
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, legend_labels,
            loc="lower center",
            ncol=len(methods),                # spread methods horizontally
            frameon=True,
            title="Methods",
            bbox_to_anchor=(0.5, -0.03)       # slightly below the plots
        )

    plt.savefig(filename, bbox_inches="tight", dpi=300)


def deviation_matrix_by_alpha_transpose(methods, labels, ground_truth, voltages, run_result, alphas, dzs, filename):
    """
    Plot a matrix of deviation curves with one subplot per (alpha, voltage) pair.
    Rows: alphas
    Columns: voltages
    Curves: methods
    """

    # Allow run_result to be either a dict or a path to a saved file
    if isinstance(run_result, (str, Path)):
        path = Path(run_result)
        print(f"Loading run results from {path} ...")
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                run_result = pickle.load(f)
        elif path.suffix == ".json":
            import json
            with open(path, "r") as f:
                run_result = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        print("Loaded successfully.")

    # Map to method names
    methods = [method.__name__ for method in methods]
    ground_truth = ground_truth.__name__

    # rows = alphas, cols = voltages; share axes
    fig, axes = plt.subplots(
        len(alphas), len(voltages),
        squeeze=False, sharex=False, sharey=True,
        figsize=(4 * len(voltages), 3 * len(alphas)),
        constrained_layout=True
    )

    # Compute scaled dzs and x-limits
    dzs = [20.6 * dz for dz in dzs]
    dz_min, dz_max = min(dzs), max(dzs)
    x_range = max(1e-9, dz_max - dz_min)
    x_pad = 0.05 * x_range
    x_left = dz_max + x_pad
    x_right = dz_min - x_pad

    for a_idx, alpha in enumerate(alphas):
        for v_idx, voltage in enumerate(voltages):
            ax = axes[a_idx, v_idx]

            # access ground truth for this voltage
            psis_gt = run_result[ground_truth][voltage][0]
            settings_gt = run_result[ground_truth][voltage][1]

            for m_idx, method in enumerate(methods):
                psis = run_result[method][voltage][0]
                settings = run_result[method][voltage][1]

                # Compute deviations for each dz at this alpha
                y_vals = []
                for d_idx, dz in enumerate(dzs):
                    res_psi = psis[a_idx][d_idx]
                    gt_psi = psis_gt[a_idx][d_idx]
                    res_settings = settings[a_idx][d_idx]
                    gt_settings = settings_gt[a_idx][d_idx]
                    dev = deviation(res_psi, res_settings, gt_psi, gt_settings)
                    y_vals.append(dev)

                ax.plot(dzs, y_vals, marker='o', markersize=3, linewidth=1.5, label=labels[m_idx])

            # Titles/labels
            ax.set_title(f"{voltage} kV at α={alpha}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_right, x_left)
            #ax.set_ylim(-1, 10)
            ax.set_yscale('log')
            ax.margins(x=0.03, y=0.10)

    # Shared axis labels (outer edges only)
    for col in range(len(voltages)):
        axes[-1, col].set_xlabel("dz")
    for row in range(len(alphas)):
        axes[row, 0].set_ylabel("Deviation from multislice")

    # Build a single shared legend using method names
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, legend_labels,
            loc="lower center",
            ncol=len(methods),
            frameon=True,
            title="Methods",
            bbox_to_anchor=(0.5, -0.03)
        )

    plt.savefig(filename, bbox_inches="tight", dpi=300)



def deviation_row_by_method_vs_voltage(methods, labels, ground_truth, voltages, run_result, alphas, dzs, filename):
    """
    Make a row of graphs (one subplot per method).
    X-axis: voltages
    Y-axis: deviation (log scale)
    Curves: alphas
    dz is fixed to 20.6 (chooses the closest available dz in your data).

    Parameters
    ----------
    methods : list[callable]
        List of method callables; their __name__ keys are used to index `run_result`.
    labels : list[str]
        Display names for the methods (same order as `methods`).
    ground_truth : callable
        Ground-truth callable; its __name__ key is used to index `run_result`.
    voltages : list
        Voltages (must match keys in `run_result[method_name]`).
    run_result : dict | str | Path
        Dict produced by your runs, or a path to a .pkl/.json with the same structure.
        Each entry is expected to be: run_result[method_name][voltage] == (psis, settings)
        where psis[alpha_idx][dz_idx] gives the needed item.
    alphas : list
        Alphas in the same order used to create psis arrays.
    dzs : list[float]
        The dz grid used in your runs (unscaled). This code matches what you did:
        scaled_dzs = [20.6 * dz for dz in dzs]
    filename : str | Path
        Where to save the figure.
    """

    # Allow run_result to be either a dict or a path to a saved file
    if isinstance(run_result, (str, Path)):
        path = Path(run_result)
        print(f"Loading run results from {path} ...")
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                run_result = pickle.load(f)
        elif path.suffix == ".json":
            import json
            with open(path, "r") as f:
                run_result = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        print("Loaded successfully.")

    # Map to method names
    method_names = [m.__name__ for m in methods]
    gt_name = ground_truth.__name__

    # Scale dzs exactly like in your original function, then pick closest to 20.6
    scaled_dzs = [20.6 * dz for dz in dzs]
    target_dz = 20.6
    dz_idx = int(np.argmin([abs(z - target_dz) for z in scaled_dzs]))
    chosen_dz = scaled_dzs[dz_idx]
    if abs(chosen_dz - target_dz) > 1e-9:
        print(f"Note: using dz≈{chosen_dz:.6g} (closest available to 20.6).")

    # Prepare figure: one row, one column per method; share Y for comparability
    fig, axes = plt.subplots(
        1, len(method_names),
        squeeze=False, sharex=False, sharey=True,
        figsize=(4 * len(method_names), 3.5),
        constrained_layout=True
    )
    axes = axes[0]  # flatten the single row

    # Ensure voltages are numeric for plotting/limits
    x_vals = list(voltages)

    for m_idx, m_name in enumerate(method_names):
        ax = axes[m_idx]

        # We'll plot one curve per alpha over voltage
        for a_idx, alpha in enumerate(alphas):
            y_vals = []
            for v in voltages:
                # Access results for this voltage
                psis_m, settings_m = run_result[m_name][v]
                psis_gt, settings_gt = run_result[gt_name][v]

                res_psi = psis_m[a_idx][dz_idx]
                gt_psi = psis_gt[a_idx][dz_idx]
                res_settings = settings_m[a_idx][dz_idx]
                gt_settings = settings_gt[a_idx][dz_idx]

                dev = deviation(res_psi, res_settings, gt_psi, gt_settings)
                y_vals.append(dev)

            ax.plot(x_vals, y_vals, marker='o', markersize=3, linewidth=1.5, label=f"α={alphas[a_idx]}")

        # Titles/labels
        ax.set_title(labels[m_idx])
        ax.grid(True, alpha=0.3)
        #ax.set_yscale('log')
        ax.set_xlabel("Voltage (kV)")
        if m_idx == 0:
            ax.set_ylabel("Deviation from multislice")
        ax.margins(x=0.03, y=0.10)
        ax.invert_xaxis()

    # Shared legend: alphas
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, legend_labels,
            loc="upper center",
            ncol=min(len(alphas), 5),
            frameon=True,
            title=f"Alphas (dz={chosen_dz:.4g})",
            bbox_to_anchor=(0.5, -0.02)
        )

    plt.savefig(filename, bbox_inches="tight", dpi=300)