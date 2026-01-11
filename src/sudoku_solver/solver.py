from pulp import *
import pandas as pd
import numpy as np
input_file = "sudoku_input.xlsx"
df_input = pd.read_excel(input_file, sheet_name="medium", skiprows=0, nrows=9, usecols="A:I", header=None)
grid_input = df_input.fillna(0).to_numpy()
R = [1, 2, 3, 4, 5, 6, 7, 8, 9]
C = [1, 2, 3, 4, 5, 6, 7, 8, 9]
D = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Boxes = {"TL": {"rows": [1,2,3], "cols": [1,2,3]}, 
         "TM": {"rows": [1,2,3], "cols": [4,5,6]}, 
         "TR": {"rows": [1,2,3], "cols": [7,8,9]},
         "ML": {"rows": [4,5,6], "cols": [1,2,3]}, 
         "MM": {"rows": [4,5,6], "cols": [4,5,6]}, 
         "MR": {"rows": [4,5,6], "cols": [7,8,9]},
         "BL": {"rows": [7,8,9], "cols": [1,2,3]}, 
         "BM": {"rows": [7,8,9], "cols": [4,5,6]}, 
         "BR": {"rows": [7,8,9], "cols": [7,8,9]} 
         }
prob = LpProblem("Sudoku_9x9", LpMinimize)
x = LpVariable.dicts("x", (R, C, D), lowBound=0, upBound=1, cat="Binary")
prob += 0
# Each cell must contain exactly one value
for r in R:
    for c in C:
        prob += lpSum(x[r][c][d] for d in D) == 1, f"OneDigitPerCell_r{r}_c{c}"
# Pre-filled cells must keep their values
for r in R:
    for c in C:
        val = grid_input[r-1, c-1]
        if val != 0: 
            prob += x[r][c][val] == 1, f"Given_r{r}_c{c}_digit{val}"
# Each value must appear exactly once in each row
for r in R:
    for d in D:
        prob += lpSum(x[r][c][d] for c in C) == 1, f"Row_r{r}_digit{d}"
# Each value must appear exactly once in each column
for c in C:
    for d in D:
        prob += lpSum(x[r][c][d] for r in R) == 1, f"Col_c{c}_digit{d}"
# Each value must appear exactly once in each 3x3 box
for box_name, box_info in Boxes.items():
    rows_in_box = box_info["rows"]
    cols_in_box = box_info["cols"]
    for d in D:
        prob += lpSum(
            x[r][c][d] for r in rows_in_box for c in cols_in_box
        ) == 1, f"Box_{box_name}_digit{d}"
solver = PULP_CBC_CMD() 
prob.solve(solver)

print("Status:", LpStatus[prob.status])
solution_grid = np.zeros((9, 9), dtype=int)
for r in R:
    for c in C:
        for d in D:
            if x[r][c][d].varValue >= 0.99:
                solution_grid[r-1][c-1] = d
print(solution_grid)
---
END
