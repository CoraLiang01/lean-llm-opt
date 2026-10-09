[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital investment budget and several strategic project dependencies (mutual exclusivity, pre-requisite, and contingent selection).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of all projects \( i \) in the project list (from 1 to 110, as given by 'Project ID').
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project \( i \) is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected net present value for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital investment required for each project.
    -   Constraint RHS (limits): The total capital investment limit is 1,000 k$ (from the query).
6.  **Formulate Objective:** Maximize the sum over all projects of ('NPV (k$)' * x[i]), i.e., maximize total expected NPV of selected projects.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of ('Capital (k$)' * x[i]) ≤ 1,000 k$.
    -   Constraint 2 (Mutually Exclusive Constraint): For Projects 4 and 7, x[4] + x[7] ≤ 1 (at most one can be selected).
    -   Constraint 3 (Pre-requisite Constraint): For Project 6 requiring Project 1, x[6] ≤ x[1] (Project 6 can only be selected if Project 1 is also selected).
    -   Constraint 4 (Contingent Constraint): For Project 10 requiring Project 5, x[10] ≤ x[5] (Project 10 can only be selected if Project 5 is also selected).
    -   Constraint 5 (Binary Selection): For all projects, x[i] ∈ {0,1}.
[Abstract Model Plan END]