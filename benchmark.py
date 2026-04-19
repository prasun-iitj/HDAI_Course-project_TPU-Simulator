
import numpy as np
import time
from systolic_array import SystolicArray
from matrix_generator import generate_matrices

def run_benchmark(size=50):
    """We compare how long CPU takes vs our simulated TPU. Shows where TPU architecture helps!"""

    print("\n" + "="*50)
    print("BENCHMARK: CPU vs TPU SIMULATOR")
    print("="*50 + "\n")
    
    print(f"Generating {size}x{size} test matrices...")
    mat_a, mat_b = generate_matrices(size)  # We generate larger matrices for real benchmarking

    # Benchmark CPU
    print(f"Testing CPU (NumPy) multiplication...")
    start = time.time()  # We start timing the CPU
    result_cpu = np.dot(mat_a, mat_b)  # We use NumPy's highly optimized dot product
    cpu_time = time.time() - start  # We measure how long it took
    print(f"✓ CPU completed in {cpu_time:.6f} seconds")

    # Benchmark TPU Simulator
    print(f"\nTesting TPU simulator multiplication...")
    tpu = SystolicArray(size)  # We create our TPU simulator

    start = time.time()  # We start timing our TPU simulator
    result_tpu = tpu.multiply(mat_a, mat_b)  # We multiply using our implementation
    tpu_time = time.time() - start  # We measure our time
    print(f"✓ TPU simulator completed in {tpu_time:.6f} seconds")

    # Verify results match
    print(f"\nVerifying results match...")
    if np.allclose(result_cpu, result_tpu):
        print("✓ Results match - Our TPU is correct!")
    else:
        print("✗ ERROR - Results don't match!")
        return

    # Display results in nice format
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Matrix size: {size}x{size}")
    print(f"CPU time: {cpu_time:.6f} seconds")
    print(f"TPU simulator time: {tpu_time:.6f} seconds")
    
    # Calculate speedup and show comparison
    if cpu_time < tpu_time:
        ratio = tpu_time / cpu_time
        print(f"\nCPU is {ratio:.1f}x faster (we're still optimizing our simulator!)")
        print("FIXME: We need to optimize our Python loops - probably too slow")
    else:
        ratio = cpu_time / tpu_time
        print(f"\nGreat! Our TPU simulator is {ratio:.1f}x faster than CPU!")
        print("(Note: This might be because NumPy overhead for this small test)")
    
    print("="*50 + "\n")

if __name__=="__main__":
    run_benchmark()
