[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies: (a) a budget constraint, (b) a mutually exclusive constraint between Projects 4 and 7, (c) a pre-requisite constraint where Project 6 can only be selected if Project 1 is also selected, and (d) a contingent constraint where Project 10 can only be selected if Project 5 is also selected. All required data is in project.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges over all 110 projects in project.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column gives the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column gives the capital investment required for each project.
    -   Constraint RHS: The total capital budget is 1,000 k$ (from the query).
    -   Project IDs and Names are used to identify specific projects for dependency constraints (e.g., Project 4, Project 7, etc.).
6.  **Formulate Objective:** Maximize the sum of selected projects' NPVs, i.e., maximize sum over all projects of `x[i] * NPV[i]`, where NPV[i] is from 'NPV (k$)'.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital invested in selected projects cannot exceed 1,000 k$. Formally, sum over all projects of `x[i] * Capital[i]` ≤ 1,000, where Capital[i] is from 'Capital (k$)'.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 (R&D Initiative Alpha) or Project 7 (Global Expansion Pilot) can be selected. Formally, `x[4] + x[7] ≤ 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 (System Automation) can only be selected if Project 1 (Infrastructure Upgrade) is also selected. Formally, `x[6] ≤ x[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 (Customer Experience Platform) can only be selected if Project 5 (Staff Training Program) is also selected. Formally, `x[10] ≤ x[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `x[i] ∈ {0,1}`.
[Abstract Model Plan END]