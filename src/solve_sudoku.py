from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import pandas as pd
import numpy as np


R = list(range(1, 10))
C = list(range(1, 10))
D = list(range(1, 10))

BOXES = {
    "TL": {"rows": [1, 2, 3], "cols": [1, 2, 3]},
    "TM": {"rows": [1, 2, 3], "cols": [4, 5, 6]},
    "TR": {"rows": [1, 2, 3], "cols": [7, 8, 9]},
    "ML": {"rows": [4, 5, 6], "cols": [1, 2, 3]},
    "MM": {"rows": [4, 5, 6], "cols": [4, 5, 6]},
    "MR": {"rows": [4, 5, 6], "cols": [7, 8, 9]},
    "BL": {"rows": [7, 8, 9], "cols": [1, 2, 3]},
    "BM": {"rows": [7, 8, 9], "cols": [4, 5, 6]},
    "BR": {"rows": [7, 8, 9], "cols": [7, 8, 9]},
}


def load_grid_from_excel(input_file: str, sheet_name: str = "medium") -> np.ndarray:
    df = pd.read_excel(
        input_file,
        sheet_name=sheet_name,
        skiprows=0,
        nrows=9,
        usecols="A:I",
        header=None,
    )
    grid = df.fillna(0).to_numpy(dtype=int)
    if grid.shape != (9, 9):
        raise ValueError(f"Expected a 9x9 grid, got {grid.shape}")
    return grid


def solve_sudoku(grid_input: np.ndarray) -> np.ndarray:
    prob = LpProblem("Sudoku_9x9", LpMinimize)
    x = LpVariable.dicts("x", (R, C, D), lowBound=0, upBound=1, cat="Binary")
    prob += 0  # feasibility problem

    # Each cell must contain exactly one digit
    for r in R:
        for c in C:
            prob += lpSum(x[r][c][d] for d in D) == 1

    # Respect givens
    for r in R:
        for c in C:
            val = int(grid_input[r - 1, c - 1])
            if val != 0:
                if val not in D:
                    raise ValueError(f"Invalid given {val} at (r={r}, c={c})")
                prob += x[r][c][val] == 1

    # Row constraints
    for r in R:
        for d in D:
            prob += lpSum(x[r][c][d] for c in C) == 1

    # Column constraints
    for c in C:
        for d in D:
            prob += lpSum(x[r][c][d] for r in R) == 1

    # Box constraints
    for box_info in BOXES.values():
        rows_in_box = box_info["rows"]
        cols_in_box = box_info["cols"]
        for d in D:
            prob += lpSum(x[r][c][d] for r in rows_in_box for c in cols_in_box) == 1

    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"No solution found. Solver status: {LpStatus[prob.status]}")

    solution = np.zeros((9, 9), dtype=int)
    for r in R:
        for c in C:
            for d in D:
                if x[r][c][d].varValue is not None and x[r][c][d].varValue >= 0.99:
                    solution[r - 1, c - 1] = d
                    break
    return solution


def main():
    input_file = "data/sudoku_input.xlsx"
    grid = load_grid_from_excel(input_file, sheet_name="medium")
    sol = solve_sudoku(grid)
    print("Solved grid:")
    print(sol)


if __name__ == "__main__":
    main()
