[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize the total expected Net Present Value (NPV), subject to a total capital investment budget and several strategic dependency constraints (mutual exclusivity, pre-requisite, and contingent selection).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem.
3.  **Define Index Sets:** The primary index is the set of all projects, indexed by `i` (where `i` ranges over all 110 projects in the project.csv file).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected net present value for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital investment required for each project.
    -   Constraint RHS: The total capital investment limit is 1,000 k$ (from the query).
    -   Project IDs and Names are used to identify specific projects for dependency constraints.
6.  **Formulate Objective:** Maximize the sum over all projects of (`NPV (k$)` * `x[i]`), i.e., maximize total expected NPV of selected projects.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of (`Capital (k$)` * `x[i]`) must be less than or equal to 1,000 k$.
    -   Constraint 2 (Mutually Exclusive Constraint): For Project 4 (R&D Initiative Alpha) and Project 7 (Global Expansion Pilot), at most one can be selected: `x[4] + x[7] <= 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 (System Automation) can only be selected if Project 1 (Infrastructure Upgrade) is also selected: `x[6] <= x[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 (Customer Experience Platform) can only be selected if Project 5 (Staff Training Program) is also selected: `x[10] <= x[5]`.
    -   Constraint 5 (Binary Selection): For all projects, `x[i]` ∈ {0,1}.
[Abstract Model Plan END]