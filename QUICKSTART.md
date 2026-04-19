# QUICKSTART Guide

## Simplest Verification for Professors / TAs

### One command to run everything:
```bash
python demo_complete.py
```
This command runs the complete project demo, showing:
- TPU systolic array concept
- matrix multiplication correctness
- CPU vs TPU performance
- cycle reduction analysis
- edge cases and requirement checks

### If you want a smaller check first:
```bash
python tpu_simulator.py
```
This runs the main TPU simulator and verifies results against NumPy.

### Run the full automated tests:
```bash
python test_simulator.py
```
This verifies correctness across multiple matrix sizes and edge cases.

## Best and simplest way to test on Google Colab

1. Open a new Google Colab notebook.
2. Upload the repository files or clone the repo.
3. Install NumPy if needed:
```python
!pip install numpy
```
4. Run the same verification commands:
```python
!python demo_complete.py
```
Optional:
```python
!python test_simulator.py
```

> The repo is pure Python and only needs NumPy, so it works on local machines and Colab without extra setup.

## Recommended professor/TA workflow

1. `python demo_complete.py` — complete project verification in one step.
2. If they want faster check: `python tpu_simulator.py`.
3. If they want validation: `python test_simulator.py`.
4. For Colab: use `!python demo_complete.py` after uploading/cloning the repo.

## Why this is easy to verify

- No hardware-specific drivers required
- No special environment beyond Python and NumPy
- The demo is self-contained and prints clear pass/fail status
- Results are directly compared to NumPy for correctness
