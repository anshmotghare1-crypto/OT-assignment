"""
=============================================================================
 BIG-M SIMPLEX METHOD
 Case Study : A classic Linear Programming Problem with MIXED constraints
              (=, >=, <=) -- the textbook example that genuinely needs
              artificial variables and the Big-M penalty technique.

 Problem (Minimization LPP):

        Minimize   Z = 4x1 + x2

        subject to:
            3x1 +  x2  = 3        ... (equality constraint -> needs artificial var)
            4x1 + 3x2 >= 6        ... (>= constraint -> needs surplus + artificial var)
             x1 + 2x2 <= 4        ... (<= constraint -> needs slack var)
             x1, x2 >= 0

 Standard form after adding slack (s), surplus (s) and artificial (A)
 variables:

        3x1 +  x2 + A1                     = 3
        4x1 + 3x2      - s1      + A2      = 6
         x1 + 2x2            + s2          = 4

 Objective (Minimize, so artificials get penalty +M):
        Z = 4x1 + x2 + 0.s1 + 0.s2 + M.A1 + M.A2

 This program builds the Big-M simplex tableau from the problem data,
 iterates automatically until optimality, and prints every iteration.
=============================================================================
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

M = 1_000_000  # a sufficiently large penalty value to represent "Big-M"


class BigMSimplex:
    def __init__(self, c, A, b, constraint_types, var_names, sense="min"):
        """
        c                : objective coefficients for the original decision variables
        A                : constraint coefficient matrix (original variables only)
        b                : RHS values (assumed made non-negative by caller)
        constraint_types : list of '<=', '>=', or '='
        var_names        : names of the original decision variables, e.g. ['x1','x2']
        sense            : 'min' or 'max'
        """
        self.sense = sense
        self.n_orig = len(var_names)
        self.orig_names = list(var_names)
        self.m = len(b)
        self.b = np.array(b, dtype=float)

        # If maximizing, internally convert to an equivalent minimization
        # problem (Big-M is presented here in its minimization form).
        self.c_orig = np.array(c, dtype=float)
        self.c_work = -self.c_orig.copy() if sense == "max" else self.c_orig.copy()

        self.A_orig = np.array(A, dtype=float)
        self.constraint_types = constraint_types

        self._build_standard_form()

    def _build_standard_form(self):
        m, n = self.m, self.n_orig
        extra_cols = []      # columns to append (slack / surplus / artificial)
        extra_costs = []     # their cost coefficients
        extra_names = []
        artificial_cols = []  # track which columns are artificial (for Phase removal)
        basis = [None] * m

        A_ext = [row[:] for row in self.A_orig.tolist()]

        col_index = n  # next free column index

        for i, ctype in enumerate(self.constraint_types):
            if ctype == "<=":
                # add a slack variable (+1), coefficient 0 in objective
                for r in range(m):
                    A_ext[r].append(1.0 if r == i else 0.0)
                extra_costs.append(0.0)
                extra_names.append(f"s{i+1}")
                artificial_cols.append(False)
                basis[i] = col_index
                col_index += 1

            elif ctype == ">=":
                # add a surplus variable (-1)
                for r in range(m):
                    A_ext[r].append(-1.0 if r == i else 0.0)
                extra_costs.append(0.0)
                extra_names.append(f"s{i+1}")
                artificial_cols.append(False)
                col_index += 1
                # add an artificial variable (+1), penalty M
                for r in range(m):
                    A_ext[r].append(1.0 if r == i else 0.0)
                extra_costs.append(M)
                extra_names.append(f"A{i+1}")
                artificial_cols.append(True)
                basis[i] = col_index
                col_index += 1

            elif ctype == "=":
                # add an artificial variable (+1), penalty M
                for r in range(m):
                    A_ext[r].append(1.0 if r == i else 0.0)
                extra_costs.append(M)
                extra_names.append(f"A{i+1}")
                artificial_cols.append(True)
                basis[i] = col_index
                col_index += 1
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")

        self.A = np.array(A_ext, dtype=float)
        self.c = np.concatenate([self.c_work, np.array(extra_costs)])
        self.var_names = self.orig_names + extra_names
        self.basis = basis
        self.artificial_flags = [False] * n + artificial_cols
        self.n_total = self.A.shape[1]

    def solve(self, verbose=True):
        A, b, c = self.A.copy(), self.b.copy(), self.c.copy()
        basis = self.basis[:]
        m, n = A.shape
        iteration = 0

        if verbose:
            print("=" * 78)
            print("INITIAL STANDARD FORM (Big-M method)")
            print("=" * 78)
            print("Variables:", self.var_names)
            print("Cost row (c_j):", c)
            print("A matrix:\n", A)
            print("b (RHS):", b)
            print("Initial basis (by column index):", basis)

        while True:
            iteration += 1
            c_B = c[basis]
            # z_j for every column, then reduced cost (c_j - z_j)
            z_j = c_B @ A
            reduced_cost = c - z_j  # for minimization, optimal when all >= 0

            if verbose:
                print("\n" + "-" * 78)
                print(f"ITERATION {iteration}")
                print("-" * 78)
                header = f"{'Basis':>6}" + "".join(f"{name:>10}" for name in self.var_names) + f"{'RHS':>12}"
                print(header)
                for i in range(m):
                    row = f"{self.var_names[basis[i]]:>6}" + "".join(f"{A[i,j]:10.3f}" for j in range(n)) + f"{b[i]:12.3f}"
                    print(row)
                zrow = f"{'Cj-Zj':>6}" + "".join(f"{reduced_cost[j]:10.3f}" for j in range(n)) + f"{c_B@b:12.3f}"
                print(zrow)

            # Optimality check (minimization): stop if all reduced costs >= 0 (within tolerance)
            if np.all(reduced_cost >= -1e-7):
                break

            # Entering variable: most negative reduced cost
            entering = np.argmin(reduced_cost)

            # Ratio test for leaving variable
            ratios = np.full(m, np.inf)
            for i in range(m):
                if A[i, entering] > 1e-9:
                    ratios[i] = b[i] / A[i, entering]

            if np.all(np.isinf(ratios)):
                raise Exception("Problem is UNBOUNDED.")

            leaving_row = np.argmin(ratios)

            if verbose:
                print(f"Entering variable : {self.var_names[entering]} "
                      f"(most negative Cj-Zj = {reduced_cost[entering]:.3f})")
                print(f"Leaving variable  : {self.var_names[basis[leaving_row]]} "
                      f"(min ratio = {ratios[leaving_row]:.3f})")

            # Pivot
            pivot = A[leaving_row, entering]
            A[leaving_row, :] = A[leaving_row, :] / pivot
            b[leaving_row] = b[leaving_row] / pivot
            for i in range(m):
                if i != leaving_row and abs(A[i, entering]) > 1e-12:
                    factor = A[i, entering]
                    A[i, :] -= factor * A[leaving_row, :]
                    b[i] -= factor * b[leaving_row]

            basis[leaving_row] = entering

            if iteration > 50:
                raise Exception("Too many iterations - check problem formulation.")

        # Check feasibility: any artificial variable still positive in basis?
        for i in range(m):
            if self.artificial_flags[basis[i]] and b[i] > 1e-6:
                raise Exception("Problem is INFEASIBLE (artificial variable remains positive).")

        # Extract solution
        solution = np.zeros(n)
        for i in range(m):
            solution[basis[i]] = b[i]

        x_values = solution[: self.n_orig]
        obj_value_work = c[: self.n_orig] @ x_values  # excludes artificials (they are 0)
        # recompute using ORIGINAL c (undo max->min flip if needed)
        obj_value = self.c_orig @ x_values

        if verbose:
            print("\n" + "=" * 78)
            print("OPTIMAL SOLUTION REACHED")
            print("=" * 78)
            for name, val in zip(self.orig_names, x_values):
                print(f"  {name} = {val:.4f}")
            print(f"  Optimal Z = {obj_value:.4f}")

        return x_values, obj_value


def main():
    print(__doc__)

    # ---- Problem data --------------------------------------------------
    # Minimize Z = 4x1 + x2
    # s.t.
    #   3x1 +  x2  = 3
    #   4x1 + 3x2 >= 6
    #    x1 + 2x2 <= 4
    #   x1, x2 >= 0
    c = [4, 1]
    A = [
        [3, 1],
        [4, 3],
        [1, 2],
    ]
    b = [3, 6, 4]
    constraint_types = ["=", ">=", "<="]
    var_names = ["x1", "x2"]

    solver = BigMSimplex(c, A, b, constraint_types, var_names, sense="min")
    x_values, obj_value = solver.solve(verbose=True)

    print("\n" + "#" * 78)
    print("FINAL ANSWER")
    print("#" * 78)
    for name, val in zip(var_names, x_values):
        print(f"  {name} = {val:.2f}")
    print(f"  Minimum Z = {obj_value:.2f}")


if __name__ == "__main__":
    main()
