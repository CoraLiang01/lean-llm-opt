[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies: a budget constraint, a mutually exclusive pair, and two prerequisite/contingent relationships.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (binary selection with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (from 1 to 110, as per 'Project ID').
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column in project.csv (maps to each project `i`).
    -   Constraint coefficients: 'Capital (k$)' column in project.csv (capital required for each project).
    -   Constraint RHS: Budget limit is 1,000 k$ (from query); logical dependencies reference specific 'Project ID' values.
6.  **Formulate Objective:** Maximize the sum over all projects of their NPV times their selection variable: maximize sum over `i` of `NPV[i] * x[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of `Capital[i] * x[i]` ≤ 1,000 k$.
    -   Constraint 2 (Mutually Exclusive Constraint): For Projects 4 and 7, `x[4] + x[7] ≤ 1` (at most one can be selected).
    -   Constraint 3 (Pre-requisite Constraint): For Project 6 requiring Project 1, `x[6] ≤ x[1]` (6 can only be selected if 1 is selected).
    -   Constraint 4 (Contingent Constraint): For Project 10 requiring Project 5, `x[10] ≤ x[5]` (10 can only be selected if 5 is selected).
    -   Constraint 5 (Binary Domain): For all projects `i`, `x[i] ∈ {0,1}`.
[Abstract Model Plan END]