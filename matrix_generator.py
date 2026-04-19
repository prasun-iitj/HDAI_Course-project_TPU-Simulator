
import numpy as np

def generate_matrices(n=3):
    """We generate two random square matrices for testing our TPU simulator. Values between 1-10 keep it simple."""
    A = np.random.randint(1,10,(n,n))  # We create matrix A with random integers
    B = np.random.randint(1,10,(n,n))  # We create matrix B with random integers
    return A,B
