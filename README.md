
# TPU Simulator Project

## What This Project Is Really About

This is our AI Hardware Design course project where we built a **simulator for how Google's Tensor Processing Units (TPUs) work**. 

We're not building actual hardware - we're simulating how it processes AI workloads efficiently using a clever structure called a **systolic array**.

### Why This Matters

Normal computers process calculations one after another (slow for AI).

TPUs use a **systolic array** - like an assembly line in a factory where data flows through multiple processing units in parallel, making AI computation much faster!

**Assembly Line Analogy:**
```
Input Data →  [ PE ][ PE ][ PE ]   = Matrix A flows horizontally
              [ PE ][ PE ][ PE ]   = Matrix B flows vertically  
              [ PE ][ PE ][ PE ]   = Data moves like assembly line
                    ↓
                  Output (Results)
```

Each **PE (Processing Element)** does one job: multiply-accumulate (C += A × B)

Data flows through the grid in waves, with multiple multiplications happening in parallel!

## What We Included

• **Systolic array simulation** - We simulate how data flows through the array cycle-by-cycle
• **Matrix multiplication** - Core operation: multiply two matrices using parallel processing
• **Performance benchmarking** - We compare CPU vs TPU simulator (shows significant speedup potential!)
• **Comprehensive testing** - We tested edge cases, error handling, and correctness
• **Educational visualization** - We show matrices and data flow patterns

## How to Run Our Project

### Quick Test (Easiest)
```bash
python tpu_simulator.py
```
- Generates test matrices, multiplies using our TPU simulator, verifies correctness

### Performance Comparison
```bash
python benchmark.py
```
- Compares execution time: NumPy CPU vs Our TPU Simulator
- Shows how parallel processing helps (even in Python!)

### Comprehensive Testing
```bash
python test_simulator.py
```
- Runs 6 test categories covering all matrix sizes and edge cases
- All tests pass! ✓

### Hardware Comparison Analysis (NEW!)
```bash
python hardware_comparison.py
```
- Shows CPU vs GPU vs TPU comparison with multiple perspectives:
  - Cycle reduction tables (O(n³) vs O(n))
  - Actual execution time measurements
  - Theoretical real-world TPU performance (1000×1000 matrices)
  - TFLOPS (trillion floating-point operations per second) comparison
  - Cost-performance analysis
  - Use case suitability analysis
  - Energy efficiency comparisons
    - Summary of what our simulator validates
- Justifies why Google built TPUs for AI workloads!

### Interactive Menu
```bash
python run.py
```
- Simple menu to choose what to run

## The Concept: Why Systolic Arrays are Clever

### Normal Matrix Multiplication (Sequential)
```
For each element C[i,j]:
  Multiply A[i] × B[j]
  Takes O(n³) operations
```

### Systolic Array (Parallel)
```
Multiple PE do calculations in parallel
Data "flows" through the array like an assembly line
Many operations happen simultaneously
Reduces effective computation cycles dramatically!
```

**Example Performance (theoretical):**
```
Matrix Size    CPU Cycles    Systolic Cycles    Speedup
4×4            64            16                  4×
8×8            512           64                  8×
```

## About This Project

We made this to understand how TPUs actually accelerate AI workloads at the hardware level.

### Key Learnings
- **Processing Elements (PEs):** Each PE multiplies two numbers and accumulates (C += A × B)
- **Data Flow Pattern:** Matrix A enters horizontally, Matrix B enters vertically
- **Parallel Processing:** Multiple PEs compute simultaneously, like an assembly line
- **Why It's Fast:** All PEs work in parallel on different data = much faster than sequential CPU

### The Architecture in Our Code
```
processing_element.py  →  Single PE (multiply-accumulate)
systolic_array.py      →  Grid of PEs (orchestrates data flow)
tpu_simulator.py       →  Entry point (runs the simulation)
benchmark.py           →  Performance comparison (CPU vs TPU)
test_simulator.py      →  Validation (tested with various matrices)
```

## Challenges We Faced

- Understanding systolic array data flow (studied TPU papers!)
- Ensuring results exactly match NumPy (validation is critical for correctness)
- Balancing educational clarity with technical accuracy
- Making it run efficiently in Python (not the most optimized, but it works!)
- Explaining the concept clearly (assembly line analogy helped!)

## Performance Results

We compared our TPU simulator with NumPy on different matrix sizes:

**Test Results (from benchmark.py):**
```
Matrix Size: 50×50
CPU Time:  0.000093 seconds
TPU Time:  0.091136 seconds
Ratio:     CPUs are 977× faster (because NumPy is highly optimized in C!)
```

**But here's the key:** On real TPUs with specialized hardware:
- The parallel PE architecture would dominate
- For very large matrices (1000×1000+), systolic advantage becomes significant
- Real TPUs use custom silicon + high parallelism

## What Makes This a Good Project

✅ Pure Python - no GPU required  
✅ Simulates real hardware concepts - systolic arrays  
✅ Demonstrates parallel processing ideas  
✅ Works in any Python environment  
✅ Educational value - understand AI hardware design  
✅ Comprehensive testing - validates correctness  

## Files Summary

**Core Simulation:**
- `processing_element.py` - Single PE multiply-accumulate logic
- `systolic_array.py` - Grid of PEs, data orchestration
- `matrix_generator.py` - Test matrix generation

**Testing & Benchmarking:**
- `tpu_simulator.py` - Main entry point with validation
- `benchmark.py` - CPU vs TPU performance comparison
- `test_simulator.py` - Comprehensive test suite (6 test categories)
- `hardware_comparison.py` - CPU vs GPU vs TPU comprehensive analysis with real-world metrics

**Utilities:**
- `run.py` - Interactive menu runner
- `visualization.py` - Matrix heatmap display
- Documentation files (DESIGN.md, INSTRUCTIONS.md, etc.)

---

## Authors & Contributions

**Group Project:** TPU Simulator for AI Hardware Design Course  
**Team:** Prasun Kumar Tripathi (M25AI2151), Kamal Kumar (M25AI2148), Vikash Kumar (M25AI2166), Sachin Yadav (M25AI2051)

*It is our course project for - "Hardware Design for AI" in the MTech program of IIT Jodhpur. We kept the code readable and honest, with a real learning focus on systolic arrays.* 🚀
