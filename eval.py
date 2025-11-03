import numpy as np

# assume input is 322x322 numpy arrays
# A is the "ground truth" AKA the Schrodinger solution 
def rel_error(A, B):
    if A.shape != B.shape:
        # error, cant compare
        return None

    # Compute the Frobenius norm of the difference
    rel_error = np.linalg.norm(A - B, ord='fro') / np.linalg.norm(A, ord='fro')

    return rel_error