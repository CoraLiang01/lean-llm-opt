[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies: (a) a budget constraint, (b) a mutually exclusive constraint between Projects 4 and 7, (c) a pre-requisite constraint where Project 6 requires Project 1, and (d) a contingent constraint where Project 10 requires Project 5.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges over all Project IDs in project.csv, i.e., 1 to 110).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital required for each project.
    -   Constraint RHS: The total capital budget is 1,000 k$ (from the query).
    -   Project IDs and Names are used to identify special constraints (e.g., Projects 1, 4, 5, 6, 7, 10).
6.  **Formulate Objective:** Maximize the sum over all projects of ('NPV (k$)' * x[i]), i.e., maximize total expected NPV of selected projects.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of ('Capital (k$)' * x[i]) ≤ 1,000 k$.
    -   Constraint 2 (Mutually Exclusive Constraint): x[4] + x[7] ≤ 1 (at most one of Project 4 or Project 7 can be selected).
    -   Constraint 3 (Pre-requisite Constraint): x[6] ≤ x[1] (Project 6 can only be selected if Project 1 is also selected).
    -   Constraint 4 (Contingent Constraint): x[10] ≤ x[5] (Project 10 can only be selected if Project 5 is also selected).
    -   Constraint 5 (Binary Domain): For all projects i, x[i] ∈ {0,1}.
[Abstract Model Plan END]