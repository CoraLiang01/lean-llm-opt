[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingent selections).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges over all Project IDs in project.csv).
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if project `i` is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital required for each project.
    -   Constraint RHS: The total capital budget is 1,000 k$ (scalar). Logical dependencies reference specific Project IDs (e.g., 4, 7, 6, 1, 10, 5).
6.  **Formulate Objective:** Maximize the sum over all projects of their expected NPV times the selection variable: maximize sum over `i` of `NPV (k$)[i] * y[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of `Capital (k$)[i] * y[i]` must not exceed 1,000 k$.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 or Project 7 can be selected: `y[4] + y[7] <= 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 can only be selected if Project 1 is also selected: `y[6] <= y[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 can only be selected if Project 5 is also selected: `y[10] <= y[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `y[i]` ∈ {0,1}.
[Abstract Model Plan END]