[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingent selections).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (binary selection with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (from all rows in project.csv, i.e., Project ID 1 to 110).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column in project.csv (maps to each project `i`).
    -   Constraint coefficients: 'Capital (k$)' column in project.csv (maps to each project `i`).
    -   Constraint RHS: Budget limit is 1,000 (k$); logical constraints reference specific Project IDs (4, 7, 6, 1, 10, 5).
6.  **Formulate Objective:** Maximize the sum over all projects of their expected NPV times the selection variable: maximize sum_i [NPV(i) * x[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of their capital requirements times the selection variable must not exceed 1,000 (k$): sum_i [Capital(i) * x[i]] ≤ 1,000.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 or Project 7 can be selected: x[4] + x[7] ≤ 1.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 can only be selected if Project 1 is selected: x[6] ≤ x[1].
    -   Constraint 4 (Contingent Constraint): Project 10 can only be selected if Project 5 is selected: x[10] ≤ x[5].
    -   Constraint 5 (Binary Domain): For all projects i, x[i] ∈ {0,1}.
[Abstract Model Plan END]