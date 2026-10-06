[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingent selections). All required data is in project.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (Project ID from 1 to 110, as listed in project.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column in project.csv (expected NPV for each project).
    -   Constraint coefficients: 'Capital (k$)' column in project.csv (capital required for each project).
    -   Constraint RHS:
        -   Budget limit: 1,000 k$ (from query).
        -   Logical constraints: project IDs and relationships as specified in the query.
6.  **Formulate Objective:** Maximize the sum of expected NPV across all selected projects, i.e., maximize sum over all projects of `NPV[i] * x[i]`, where `NPV[i]` is from the 'NPV (k$)' column.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital invested in selected projects cannot exceed 1,000 k$. Formally, sum over all projects of `Capital[i] * x[i]` ≤ 1,000, where `Capital[i]` is from the 'Capital (k$)' column.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 (R&D Initiative Alpha) or Project 7 (Global Expansion Pilot) can be selected. Formally, `x[4] + x[7] ≤ 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 (System Automation) can only be selected if Project 1 (Infrastructure Upgrade) is also selected. Formally, `x[6] ≤ x[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 (Customer Experience Platform) can only be selected if Project 5 (Staff Training Program) is also selected. Formally, `x[10] ≤ x[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `x[i] ∈ {0,1}`.
[Abstract Model Plan END]