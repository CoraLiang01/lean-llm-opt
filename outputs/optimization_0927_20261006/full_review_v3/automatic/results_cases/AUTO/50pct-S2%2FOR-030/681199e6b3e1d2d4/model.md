[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingencies), using data from project.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (binary selection with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges over all Project IDs in project.csv).
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital required for each project.
    -   Constraint RHS: Budget limit is 1,000 k$ (from query); logical dependencies are defined by Project IDs and relationships described in the query.
6.  **Formulate Objective:** Maximize the sum over all projects of (NPV (k$) * y[i]); that is, maximize total expected NPV of selected projects.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of (Capital (k$) * y[i]) ≤ 1,000.
    -   Constraint 2 (Mutually Exclusive Constraint): y[4] + y[7] ≤ 1 (at most one of Project 4 or Project 7 can be selected).
    -   Constraint 3 (Pre-requisite Constraint): y[6] ≤ y[1] (Project 6 can only be selected if Project 1 is also selected).
    -   Constraint 4 (Contingent Constraint): y[10] ≤ y[5] (Project 10 can only be selected if Project 5 is also selected).
    -   Constraint 5 (Binary Domain): y[i] ∈ {0,1} for all projects i.
[Abstract Model Plan END]