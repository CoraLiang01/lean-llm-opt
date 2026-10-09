Let x_j = number of units of HiFi-j to produce per day, for j = 1,...,101.

Parameters (from workstation_times.csv, preserving file and row order):

Let p_{i,j} = processing time (in minutes) required per unit of HiFi-j at workstation i, for i = 1,2,3 and j = 1,...,101, as given below:

- For Workstation 1 (i=1): p_{1,1} = 6, p_{1,2} = 4, ..., p_{1,101} = 9
- For Workstation 2 (i=2): p_{2,1} = 5, p_{2,2} = 5, ..., p_{2,101} = 3
- For Workstation 3 (i=3): p_{3,1} = 4, p_{3,2} = 6, ..., p_{3,101} = 6

Effective capacities:
- C_1 = 1,296
- C_2 = 1,238.4
- C_3 = 1,267.2

Variables:
- x_j ∈ {0, 1, 2, ...} for j = 1,...,101

Objective:
Minimize total idle time across all workstations:
\[
\text{Minimize} \quad \sum_{i=1}^3 \left[ C_i - \sum_{j=1}^{101} p_{i,j} x_j \right]
\]
which is equivalent to:
\[
\text{Minimize} \quad (C_1 + C_2 + C_3) - \sum_{i=1}^3 \sum_{j=1}^{101} p_{i,j} x_j
\]
or, equivalently, maximizing total processing time used (since the constant term is fixed):
\[
\text{Minimize} \quad 3,801.6 - \sum_{i=1}^3 \sum_{j=1}^{101} p_{i,j} x_j
\]

Subject to:
\[
\sum_{j=1}^{101} p_{1,j} x_j \leq 1,296
\]
\[
\sum_{j=1}^{101} p_{2,j} x_j \leq 1,238.4
\]
\[
\sum_{j=1}^{101} p_{3,j} x_j \leq 1,267.2
\]
\[
x_j \geq 0,\quad x_j \in \mathbb{Z} \quad \forall j = 1,\ldots,101
\]

Where the p_{i,j} coefficients are as follows (from the CSV, in original order):

Workstation 1 (row 1): [6, 4, 6, 7, 6, 6, 8, 9, 6, 7, 1, 2, 4, 7, 3, 8, 3, 2, 4, 5, 8, 3, 2, 3, 9, 7, 3, 5, 7, 6, 2, 1, 5, 6, 5, 1, 7, 9, 8, 3, 3, 8, 2, 3, 3, 8, 9, 2, 3, 4, 2, 9, 2, 1, 8, 8, 4, 4, 6, 1, 6, 5, 3, 5, 1, 6, 6, 5, 3, 4, 3, 8, 1, 2, 3, 2, 8, 4, 4, 2, 7, 5, 1, 6, 4, 1, 3, 8, 3, 3, 3, 3, 6, 7, 6, 2, 1, 8, 9, 7, 9]
Workstation 2 (row 2): [5, 5, 5, 1, 7, 8, 7, 5, 6, 8, 9, 9, 2, 6, 9, 4, 1, 2, 9, 3, 8, 5, 9, 5, 8, 7, 1, 1, 9, 7, 1, 9, 6, 4, 7, 4, 8, 6, 5, 3, 6, 7, 6, 2, 1, 1, 3, 8, 4, 3, 6, 9, 8, 7, 2, 2, 5, 4, 3, 8, 8, 6, 6, 3, 1, 6, 2, 6, 1, 3, 7, 1, 1, 2, 8, 7, 8, 8, 7, 5, 2, 5, 6, 2, 3, 2, 3, 8, 4, 9, 6, 1, 4, 8, 8, 6, 8, 5, 5, 8, 3]
Workstation 3 (row 3): [4, 6, 5, 2, 6, 5, 3, 3, 4, 8, 6, 3, 3, 3, 7, 8, 3, 8, 1, 5, 3, 8, 5, 8, 4, 8, 6, 7, 9, 5, 3, 6, 3, 3, 3, 8, 4, 6, 3, 8, 3, 7, 5, 3, 1, 8, 9, 6, 6, 4, 7, 1, 9, 9, 3, 9, 6, 5, 7, 8, 9, 9, 8, 5, 4, 4, 3, 3, 8, 8, 2, 4, 9, 6, 7, 6, 7, 3, 1, 7, 6, 4, 3, 5, 7, 6, 3, 5, 2, 2, 9, 3, 6, 9, 7, 2, 4, 5, 8, 1, 6]

All data is preserved in original file and row order.

This is a complete integer programming formulation for the problem as described.