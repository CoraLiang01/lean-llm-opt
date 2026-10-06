[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize the total expected Net Present Value (NPV), subject to a total capital investment budget and several strategic project dependencies (mutual exclusivity, pre-requisite, and contingent selection).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by Project ID (from 1 to 110).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project i is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected net present value for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital investment required for each project.
    -   Constraint RHS (limits): The total capital investment limit is 1,000 k$ (given in the query).
    -   Project dependencies: Specific project IDs and their relationships are defined in the query (e.g., Project 4 & 7 are mutually exclusive, Project 6 requires 1, Project 10 requires 5).
6.  **Formulate Objective:** Maximize the sum of selected projects' NPVs, i.e., maximize sum over all projects of `NPV (k$)[i] * x[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum of capital investments for selected projects must not exceed 1,000 k$, i.e., sum over all projects of `Capital (k$)[i] * x[i] <= 1,000`.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 (R&D Initiative Alpha) or Project 7 (Global Expansion Pilot) can be selected, i.e., `x[4] + x[7] <= 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 (System Automation) can only be selected if Project 1 (Infrastructure Upgrade) is also selected, i.e., `x[6] <= x[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 (Customer Experience Platform) can only be selected if Project 5 (Staff Training Program) is also selected, i.e., `x[10] <= x[5]`.
    -   (No additional constraints are specified; all other projects are independent.)
[Abstract Model Plan END]