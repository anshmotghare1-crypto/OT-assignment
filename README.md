# Optimization Techniques — Assignment 1

Python implementations of two classic Operations Research techniques, built
from first principles (no external LP/transportation solver libraries used
for the core algorithms — only NumPy for matrix arithmetic).

## 1. Big-M Simplex Method (`big_m_method.py`)

Solves the LPP:

```
Minimize   Z = 4x1 + x2
subject to:
    3x1 +  x2  = 3
    4x1 + 3x2 >= 6
     x1 + 2x2 <= 4
    x1, x2 >= 0
```

This problem was deliberately chosen because its mixed constraint types
(`=`, `>=`, `<=`) require slack, surplus, **and** artificial variables,
making it a genuine test of the Big-M penalty method.

The script:
- Automatically converts the LPP to standard form (adding slack/surplus/
  artificial variables as needed per constraint type).
- Assigns a Big-M penalty cost to artificial variables.
- Runs the simplex algorithm iteration by iteration, printing the full
  tableau, entering/leaving variables, and ratio test at each step.
- Reports the optimal decision variable values and objective value.

**Result:** x1 = 0.40, x2 = 1.80, Minimum Z = 3.40
(cross-checked against `scipy.optimize.linprog`).

Run it:
```bash
python3 big_m_method.py
```

## 2. Transportation Problem — VAM + MODI (`transportation_vam_modi.py`)

Solves a balanced transportation problem with 3 sources and 4 destinations:

|        | D1 | D2 | D3 | D4 | Supply |
|--------|----|----|----|----|--------|
| S1     | 19 | 30 | 50 | 10 | 7      |
| S2     | 70 | 30 | 40 | 60 | 9      |
| S3     | 40 | 8  | 70 | 20 | 18     |
| Demand | 5  | 8  | 7  | 14 | 34     |

The script:
- **Step 1 — Vogel's Approximation Method (VAM):** computes row/column
  penalties, selects allocation cells, and builds an Initial Basic
  Feasible Solution (IBFS).
- **Step 2 — MODI Method:** computes the dual potentials `u_i`, `v_j`,
  tests optimality via opportunity costs `Δ_ij = c_ij − (u_i + v_j)`,
  finds a closed loop for any improving cell, and reallocates flow along
  the loop. Repeats until all opportunity costs are non-negative.

**Result:**
- IBFS cost (VAM) = 779
- Optimal cost (after MODI) = **743**
- Optimal plan: S1→D1: 5, S1→D4: 2, S2→D2: 2, S2→D3: 7, S3→D2: 6, S3→D4: 12

(cross-checked against a `scipy.optimize.linprog` formulation of the same
transportation problem.)

Run it:
```bash
python3 transportation_vam_modi.py
```

## Files

| File | Description |
|------|-------------|
| `big_m_method.py` | Big-M Simplex Method implementation |
| `transportation_vam_modi.py` | VAM + MODI transportation solver |
| `big_m_output.txt` | Captured console output for Big-M run |
| `transport_output.txt` | Captured console output for VAM+MODI run |
| `Assignment1_OptimizationTechniques.pdf` | Submission PDF with code & output |

## Requirements

```bash
pip install numpy
```
(SciPy is only used for independent verification, not for the core algorithms.)
