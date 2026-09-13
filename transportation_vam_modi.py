"""
=============================================================================
 TRANSPORTATION PROBLEM
 Case Study : A classic balanced transportation problem with
              3 sources (factories/plants) and 4 destinations (warehouses).

              Step 1 -> Vogel's Approximation Method (VAM)
                        to obtain an Initial Basic Feasible Solution (IBFS).

              Step 2 -> MODI (Modified Distribution) Method
                        to test the IBFS for optimality and iteratively
                        improve it (using u_i, v_j potentials and the
                        closed-loop reallocation / stepping-stone rule)
                        until the minimum-cost optimal shipment plan
                        is reached.

 Problem data:

                    D1    D2    D3    D4   | Supply
              S1     19    30    50    10  |   7
              S2     70    30    40    60  |   9
              S3     40     8    70    20  |  18
             -----------------------------------------
             Demand   5     8     7    14  |  34   (Total supply = Total demand -> balanced)

 (This is the well-known textbook transportation example used to
  illustrate VAM + MODI.)
=============================================================================
"""

import numpy as np
import copy

np.set_printoptions(precision=2, suppress=True)

INF = float("inf")


# =============================================================================
# STEP 1: VOGEL'S APPROXIMATION METHOD (VAM)  -- Initial Basic Feasible Solution
# =============================================================================
def vogel_approximation_method(cost, supply, demand, verbose=True):
    cost = np.array(cost, dtype=float)
    supply = list(map(float, supply))
    demand = list(map(float, demand))
    m, n = cost.shape

    allocation = np.zeros((m, n))
    row_done = [False] * m
    col_done = [False] * n

    step = 0
    if verbose:
        print("=" * 78)
        print("STEP 1: VOGEL'S APPROXIMATION METHOD (VAM) -> Initial Basic Feasible Solution")
        print("=" * 78)

    while (not all(row_done)) and (not all(col_done)):
        step += 1

        # ---- compute row penalties -----------------------------------
        row_penalty = []
        for i in range(m):
            if row_done[i]:
                row_penalty.append(-1)
                continue
            available = [cost[i, j] for j in range(n) if not col_done[j]]
            if len(available) >= 2:
                available.sort()
                row_penalty.append(available[1] - available[0])
            elif len(available) == 1:
                row_penalty.append(available[0])
            else:
                row_penalty.append(-1)

        # ---- compute column penalties ----------------------------------
        col_penalty = []
        for j in range(n):
            if col_done[j]:
                col_penalty.append(-1)
                continue
            available = [cost[i, j] for i in range(m) if not row_done[i]]
            if len(available) >= 2:
                available.sort()
                col_penalty.append(available[1] - available[0])
            elif len(available) == 1:
                col_penalty.append(available[0])
            else:
                col_penalty.append(-1)

        max_row_pen = max(row_penalty)
        max_col_pen = max(col_penalty)

        if verbose:
            print(f"\n--- VAM Step {step} ---")
            print("Row penalties   :", [f"{p:.0f}" if p >= 0 else "-" for p in row_penalty])
            print("Column penalties:", [f"{p:.0f}" if p >= 0 else "-" for p in col_penalty])

        if max_row_pen >= max_col_pen:
            i = row_penalty.index(max_row_pen)
            # choose min-cost cell in that row among open columns
            j = min((j for j in range(n) if not col_done[j]), key=lambda jj: cost[i, jj])
        else:
            j = col_penalty.index(max_col_pen)
            i = min((i for i in range(m) if not row_done[i]), key=lambda ii: cost[ii, j])

        qty = min(supply[i], demand[j])
        allocation[i, j] = qty
        supply[i] -= qty
        demand[j] -= qty

        if verbose:
            print(f"Selected cell (S{i+1}, D{j+1}) with cost {cost[i,j]:.0f} "
                  f"-> allocate {qty:.0f} units")

        if abs(supply[i]) < 1e-9:
            row_done[i] = True
        if abs(demand[j]) < 1e-9:
            col_done[j] = True

    if verbose:
        print("\nInitial Basic Feasible Solution (allocation matrix):")
        print(allocation)
        total_cost = float(np.sum(allocation * cost))
        print(f"Total cost of IBFS (via VAM) = {total_cost:.2f}")

    return allocation


# =============================================================================
# STEP 2: MODI (MODIFIED DISTRIBUTION) METHOD -- Optimality test & improvement
# =============================================================================
def find_closed_loop(allocation, start):
    """Find a closed loop of basic cells starting/ending at `start`,
    used to reallocate flow when an improving (non-basic) cell enters
    the basis. Returns list of (row, col) cells forming the loop
    (alternating +,-,+,-,...)."""
    m, n = allocation.shape
    basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 1e-9]
    basic_cells.append(start)

    def get_row_cells(r, exclude):
        return [c for c in basic_cells if c[0] == r and c != exclude]

    def get_col_cells(c_, exclude):
        return [c for c in basic_cells if c[1] == c_ and c != exclude]

    def search(path, turn):
        # turn: 'row' means we must move along the same row next, 'col' same col
        last = path[-1]
        if len(path) > 3 and last == start:
            return path
        candidates = get_row_cells(last[0], last) if turn == "row" else get_col_cells(last[1], last)
        for nxt in candidates:
            if nxt == start and len(path) >= 3:
                return path + [start]
            if nxt in path:
                continue
            result = search(path + [nxt], "col" if turn == "row" else "row")
            if result:
                return result
        return None

    loop = search([start], "row")
    if loop is None:
        loop = search([start], "col")
    return loop


def modi_method(cost, allocation, supply, demand, verbose=True):
    cost = np.array(cost, dtype=float)
    allocation = allocation.copy()
    m, n = cost.shape

    if verbose:
        print("\n" + "=" * 78)
        print("STEP 2: MODI (MODIFIED DISTRIBUTION) METHOD -> Optimality test")
        print("=" * 78)

    iteration = 0
    while True:
        iteration += 1

        # ---- degeneracy check: need exactly m+n-1 basic cells ----------
        basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 1e-9]
        needed = m + n - 1
        if len(basic_cells) < needed:
            # resolve degeneracy: add an epsilon allocation (0+) at the
            # least-cost independent cell so u_i, v_j can be computed
            eps = 1e-6
            for i in range(m):
                for j in range(n):
                    if allocation[i, j] <= 1e-9:
                        allocation[i, j] = eps
                        basic_cells = [(a, b) for a in range(m) for b in range(n) if allocation[a, b] > 1e-9]
                        if len(basic_cells) == needed:
                            break
                if len(basic_cells) == needed:
                    break

        # ---- compute u_i, v_j potentials -------------------------------
        u = [None] * m
        v = [None] * n
        u[0] = 0
        changed = True
        while changed:
            changed = False
            for (i, j) in basic_cells:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i, j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i, j] - v[j]
                    changed = True

        if verbose:
            print(f"\n--- MODI Iteration {iteration} ---")
            print("Basic cells (i,j):", basic_cells)
            print("u_i =", ["-" if x is None else round(x, 2) for x in u])
            print("v_j =", ["-" if x is None else round(x, 2) for x in v])

        # ---- compute opportunity cost (delta_ij = c_ij - u_i - v_j) for
        #      non-basic cells ------------------------------------------
        delta = np.zeros((m, n))
        most_negative = 0.0
        entering_cell = None
        for i in range(m):
            for j in range(n):
                if (i, j) not in basic_cells:
                    delta[i, j] = cost[i, j] - (u[i] + v[j])
                    if delta[i, j] < most_negative:
                        most_negative = delta[i, j]
                        entering_cell = (i, j)

        if verbose:
            print("Opportunity costs (delta_ij) for non-basic cells:")
            print(delta)

        if entering_cell is None:
            if verbose:
                print("\nAll opportunity costs >= 0  ->  OPTIMAL SOLUTION REACHED.")
            break

        if verbose:
            print(f"Most negative opportunity cost at cell {entering_cell} "
                  f"= {most_negative:.2f}  -> this cell enters the basis.")

        # ---- build the closed loop and reallocate ----------------------
        loop = find_closed_loop(allocation, entering_cell)
        if loop is None:
            raise Exception("Could not find a closed loop - check degeneracy handling.")

        loop = loop[:-1]  # drop the duplicated closing cell
        # alternate signs starting with + at entering cell
        minus_cells = loop[1::2]
        theta = min(allocation[i, j] for (i, j) in minus_cells)

        if verbose:
            signs = ["+" if k % 2 == 0 else "-" for k in range(len(loop))]
            print("Closed loop:", list(zip(loop, signs)))
            print(f"theta (min allocation among '-' cells) = {theta:.2f}")

        for k, (i, j) in enumerate(loop):
            if k % 2 == 0:
                allocation[i, j] += theta
            else:
                allocation[i, j] -= theta

        # clean near-zero allocations (including epsilon degeneracy patches)
        allocation[np.abs(allocation) < 1e-5] = 0.0

        if verbose:
            print("Updated allocation matrix:")
            print(allocation)

        if iteration > 50:
            raise Exception("Too many MODI iterations - check problem data.")

    total_cost = float(np.sum(allocation * cost))
    return allocation, total_cost


def print_allocation_table(allocation, cost, supply, demand):
    m, n = cost.shape
    print("\nFinal optimal shipment plan:")
    header = "        " + "".join(f"D{j+1:<8}" for j in range(n)) + "Supply"
    print(header)
    for i in range(m):
        row = f"S{i+1}      " + "".join(f"{allocation[i,j]:<9.1f}" for j in range(n)) + f"{supply[i]:<6.0f}"
        print(row)
    demrow = "Demand  " + "".join(f"{d:<9.0f}" for d in demand)
    print(demrow)


def main():
    print(__doc__)

    cost = [
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40, 8, 70, 20],
    ]
    supply = [7, 9, 18]
    demand = [5, 8, 7, 14]

    assert sum(supply) == sum(demand), "Problem must be balanced (total supply = total demand)."

    # ---- Step 1: VAM for Initial Basic Feasible Solution -------------------
    ibfs = vogel_approximation_method(cost, supply[:], demand[:], verbose=True)

    # ---- Step 2: MODI method for optimality test & improvement ------------
    optimal_allocation, optimal_cost = modi_method(cost, ibfs, supply, demand, verbose=True)

    print("\n" + "#" * 78)
    print("FINAL RESULT")
    print("#" * 78)
    print_allocation_table(optimal_allocation, np.array(cost, dtype=float), supply, demand)
    print(f"\nMinimum Total Transportation Cost = {optimal_cost:.2f}")


if __name__ == "__main__":
    main()
