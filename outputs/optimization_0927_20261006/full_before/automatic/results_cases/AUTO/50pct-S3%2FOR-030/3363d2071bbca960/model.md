[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingent selections).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges over all 110 projects in project.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected for investment, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column gives the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column gives the capital required for each project.
    -   Constraint RHS: Budget limit is 1,000 k$ (from the query, not the CSV).
    -   Project IDs and Names are used to identify special constraints (e.g., Project 4, 7, 6, 1, 10, 5).
6.  **Formulate Objective:** Maximize the sum of selected projects' NPVs, i.e., maximize sum over all projects of `x[i] * NPV (k$)[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital invested in selected projects cannot exceed 1,000 k$. Formally, sum over all projects of `x[i] * Capital (k$)[i] <= 1,000`.
    -   Constraint 2 (Mutually Exclusive Projects 4 & 7): At most one of Project 4 (R&D Initiative Alpha) or Project 7 (Global Expansion Pilot) can be selected: `x[4] + x[7] <= 1`.
    -   Constraint 3 (Pre-requisite: Project 6 requires 1): Project 6 (System Automation) can only be selected if Project 1 (Infrastructure Upgrade) is also selected: `x[6] <= x[1]`.
    -   Constraint 4 (Contingent: Project 10 requires 5): Project 10 (Customer Experience Platform) can only be selected if Project 5 (Staff Training Program) is also selected: `x[10] <= x[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `x[i]` is binary (0 or 1).
[Abstract Model Plan END]