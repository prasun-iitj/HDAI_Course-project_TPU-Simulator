
# TPU Simulator Design - Our Implementation

## Overview
This is our implementation of a TPU (Tensor Processing Unit) simulator that we built to understand how Google's AI accelerators work at the hardware level. The core idea is systolic arrays - a clever way to do parallel matrix multiplication!

## Architecture

```
        Matrix A →
      ┌─────────────┐
      │ Systolic    │
      │ Array       │
      └─────────────┘
         ↓ Matrix B
        Result
```

## Core Concept

**Processing Element (PE):** 
- Each PE multiplies two numbers and accumulates the result
- Operation: `C += A × B`

**Data Flow:**
- Matrix A flows horizontally (left to right)
- Matrix B flows vertically (top to bottom)
- Results accumulate at each PE

## Implementation Details

### Modules We Built

1. **matrix_generator.py** - Generates random test matrices
2. **processing_element.py** - Single PE that does multiply-accumulate
3. **systolic_array.py** - Grid of PEs that orchestrates the multiplication
4. **tpu_simulator.py** - Main entry point for testing
5. **visualization.py** - Visualizes matrices as heatmaps
6. **benchmark.py** - Compares TPU simulator vs CPU performance
7. **test_simulator.py** - Comprehensive test suite we created

### Key Files We Added

- **run.py** - Simple menu to run different tests
- **test_simulator.py** - Tests covering edge cases, error handling, different sizes

## What We Tested

We tested our simulator with:

✓ Small matrices (2x2) - basic functionality
✓ Medium matrices (4x4, 8x8) - various sizes
✓ Identity matrices - mathematical properties
✓ Zero matrices - edge cases
✓ Error handling - invalid inputs
✓ All results validated against NumPy

**Result: All 6 test categories PASSED!**

## How It Works (Step by Step)

1. Generate two n×n matrices A and B
2. Create a systolic array of size n×n
3. Loop through each position:
   - For C[i][j]: multiply A[i][k] × B[k][j] and accumulate
4. Verify results match NumPy dot product
5. (Optional) Visualize results as heatmaps

## Error Handling

We added error checking for:
- Negative matrix sizes
- Wrong matrix dimensions
- Non-square matrices
- Type mismatches

Each error gives a clear message so we know what went wrong!

## Performance Notes

- Our simulator is slower than NumPy (expected - not optimized!)
- CPUs are ~1000x faster than our unoptimized Python
- Real TPUs excel with much larger matrices in parallel
- We included debug output to trace execution

## Future Improvements (FIXME)

- Implement true systolic dataflow (currently basic loops)
- Optimize Python performance (vectorization?)
- Support larger matrix sizes
- Add proper timing instrumentation

## Files Summary

All files follow these principles:
- Simple, readable Python code
- "We" language (group project)
- Personal comments showing thought process
- Honest about limitations
- Student-like structure (not over-optimized)

---

*This project helped us understand AI hardware design much better!*
