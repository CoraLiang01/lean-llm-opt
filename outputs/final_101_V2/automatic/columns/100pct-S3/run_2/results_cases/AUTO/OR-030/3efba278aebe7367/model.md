[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingent selections). All required data is in project.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges from 1 to 110, corresponding to 'Project ID').
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if project `i` is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column provides the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column provides the capital required for each project.
    -   Constraint RHS: The total capital budget is 1,000 k$ (scalar). Logical constraints reference specific project IDs (e.g., 4, 7, 6, 1, 10, 5).
6.  **Formulate Objective:** Maximize the sum of selected projects' NPVs, i.e., maximize sum over all projects of `y[i] * NPV[i]`, where NPV[i] is from 'NPV (k$)'.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital invested in selected projects cannot exceed 1,000 k$. Formally, sum over all projects of `y[i] * Capital[i]` ≤ 1,000, where Capital[i] is from 'Capital (k$)'.
    -   Constraint 2 (Mutually Exclusive Projects 4 & 7): At most one of Project 4 or Project 7 can be selected. Formally, `y[4] + y[7] ≤ 1`.
    -   Constraint 3 (Pre-requisite: Project 6 requires 1): Project 6 can only be selected if Project 1 is also selected. Formally, `y[6] ≤ y[1]`.
    -   Constraint 4 (Contingent: Project 10 requires 5): Project 10 can only be selected if Project 5 is also selected. Formally, `y[10] ≤ y[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `y[i] ∈ {0,1}`.
[Abstract Model Plan END]