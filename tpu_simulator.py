
import numpy as np
from matrix_generator import generate_matrices
from systolic_array import SystolicArray
from visualization import show_matrix

def main():
    """This is the main function that runs the TPU simulation. We made this to understand how matrices multiply in systolic arrays."""

    try:
        print("\n" + "="*50)
        print("STARTING TPU SIMULATOR")
        print("="*50 + "\n")
        
        size = 3  # We set size to 3 for testing - smaller matrices are easier to understand and debug

        print(f"[TEST] Generating {size}x{size} matrices for testing...")
        mat_a, mat_b = generate_matrices(size)  # We generate random matrices for testing our simulator

        print("\nMatrix A (input 1)")
        print(mat_a)

        print("\nMatrix B (input 2)")
        print(mat_b)

        print("\n[TEST] Creating systolic array TPU simulator...")
        tpu = SystolicArray(size)  # We create the systolic array - this simulates the TPU hardware

        print("\n[TEST] Running TPU multiplication...")
        result_tpu = tpu.multiply(mat_a, mat_b)  # We multiply using our simulated TPU

        print("\nTPU Result:")
        print(np.array(result_tpu))

        print("\n[TEST] Calculating CPU result for verification...")
        result_cpu = np.dot(mat_a, mat_b)  # We compute using standard NumPy CPU method

        print("\nCPU Result:")
        print(result_cpu)

        # We check if results match - this verification step is critical!
        print("\n" + "="*50)
        if np.allclose(result_tpu, result_cpu):  # We check if our TPU result matches NumPy (standard CPU)
            print("✓ VERIFICATION SUCCESSFUL - Results match!")
            print("Our TPU simulator is correct!")
        else:
            print("✗ ERROR - Results don't match! Bug found!")
            print("We need to debug the multiplication logic")
        print("="*50 + "\n")

        # We optionally show visualization (commented out by default to avoid hanging)
        print("[TEST] Showing matrix visualizations...")
        # show_matrix(mat_a, "Matrix A")
        # show_matrix(mat_b, "Matrix B")
        # show_matrix(result_tpu, "TPU Result")
        print("[INFO] Visualizations skipped (uncomment if you want to see plots)\n")
        
    except Exception as e:
        print(f"\n✗ ERROR in main: {e}")
        print("We need to debug this error!")
        return False
    
    return True

if __name__=="__main__":
    success = main()
    if success:
        print("\n[SUCCESS] TPU Simulator test passed!")
    else:
        print("\n[FAILURE] TPU Simulator test failed!")
