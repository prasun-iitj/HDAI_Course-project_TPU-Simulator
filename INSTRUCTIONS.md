
# Project Instructions & Setup

## Requirements

- **Python:** 3.8 or higher
- **Required Libraries:**
  - numpy (matrix operations)
  - matplotlib (visualization)
  - time (benchmarking)

## Installation

```bash
# Install requirements
pip install numpy matplotlib
```

## How to Run Our Project

We made it simple - you have multiple ways to run our simulator:

### Option 1: Simple Menu (Easiest)
```bash
python run.py
```
This shows you a menu to choose what to run!

### Option 2: Run Specific Scripts

**Main Simulator:**
```bash
python tpu_simulator.py
```
Multiplies two 3×3 matrices and verifies the result matches NumPy.

**Benchmarking:**
```bash
python benchmark.py
```
Compares CPU speed vs our TPU simulator on 50×50 matrices.

**Comprehensive Testing:**
```bash
python test_simulator.py
```
Runs 6 different test categories:
- Test 1: Small 2×2 matrices
- Test 2: Medium 4×4 matrices
- Test 3: Identity matrices
- Test 4: Zero matrices
- Test 5: Larger 8×8 matrices
- Test 6: Error handling

## Project Structure

```
TPU-Simulator/
├── matrix_generator.py    - Generate test matrices
├── processing_element.py   - Single PE logic
├── systolic_array.py       - TPU simulator core
├── tpu_simulator.py        - Main entry point
├── visualization.py        - Matrix visualization
├── benchmark.py            - Performance comparison
├── test_simulator.py       - Test suite (we added this!)
├── run.py                  - Menu runner (we added this!)
├── DESIGN.md              - Architecture details
├── INSTRUCTIONS.md        - This file
├── README.md              - Project overview
└── (other utility files)
```

## Key Rules We Followed

**Processing Element Operation:**
- Each PE does: `result += input_a × input_b`
- This is the accumulation that makes systolic arrays work!

**Validation:**
- All results verified with `numpy.dot(A,B)`
- We must match NumPy exactly
- Debug output helps trace any issues

**Matrix Flow:**
- Matrix A flows horizontally (left to right)
- Matrix B flows vertically (top to bottom)
- Data propagates like waves through the array (systolic flow)

## What Each Module Does

1. **matrix_generator.py**
   - Creates random test matrices
   - Simple values (1-9) for easy debugging

2. **processing_element.py**
   - Represents one PE in the systolic array
   - Stores accumulator value
   - Does multiply-accumulate operation

3. **systolic_array.py** ⭐ MAIN TPU SIMULATOR
   - Creates grid of PEs
   - Implements matrix multiplication
   - Validates input matrices
   - Returns result matrix

4. **tpu_simulator.py** ⭐ MAIN ENTRY POINT
   - Generates test matrices
   - Creates TPU simulator
   - Runs multiplication
   - Verifies results
   - Shows visualizations

5. **visualization.py**
   - Displays matrices as heatmaps
   - Helps understand data patterns

6. **benchmark.py**
   - Measures execution time for CPU vs TPU
   - Compares 50×50 matrices
   - Shows speedup calculations

7. **test_simulator.py** 🆕 WE ADDED
   - Comprehensive test suite
   - Tests various matrix sizes
   - Tests edge cases
   - Tests error handling
   - Shows detailed results

8. **run.py** 🆕 WE ADDED
   - Simple menu for running tests
   - User-friendly interface

## Testing Strategy

We thoroughly tested our simulator:

**Test Coverage:**
- ✓ Matrix sizes: 2×2, 4×4, 8×8
- ✓ Special cases: identity, zero matrices
- ✓ Error conditions: invalid sizes, dimensions
- ✓ Result verification: vs NumPy

**Result:**
- 6/6 test categories PASSED
- All edge cases handled
- Error messages clear and helpful

## Debugging Tips

If something goes wrong:

1. Check matrix dimensions match
2. Look at debug output (lines starting with `[DEBUG]`)
3. Run test_simulator.py to isolate the issue
4. Verify matrices are square (n×n)
5. Check for type errors (must be integers/floats)

## Performance Expectations

**Simulation Times (approximate):**
- 2×2: < 1ms
- 4×4: < 1ms
- 8×8: < 10ms
- 50×50: ~90ms (not optimized!)

**Comparison:**
- NumPy is ~1000x faster (highly optimized C code)
- Our simulator is pure Python (educational, not optimized)

## Notes

- This is an educational simulator, not production-grade
- We focused on correctness and understanding over performance
- All code designed to be readable and learnable
- Comments explain the "why" not just the "what"

## Authors

Group Project: TPU Simulator for AI Hardware Design Course  
Team: Prasun Kumar Tripathi (and team members)

---

**Questions?** Check DESIGN.md for architecture details or look at the code comments!
