[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic dependency constraints: (a) total capital investment ≤ 1,000 k$; (b) at most one of Projects 4 or 7 can be selected; (c) Project 6 can only be selected if Project 1 is also selected; (d) Project 10 can only be selected if Project 5 is also selected. All required data is in project.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by i ∈ {1, ..., 110}, corresponding to 'Project ID' in project.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project i is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column gives the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column gives the capital required for each project.
    -   Constraint RHS: Budget limit is 1,000 k$ (from query); logical dependencies reference specific 'Project ID' values (e.g., 4, 7, 6, 1, 10, 5).
6.  **Formulate Objective:** Maximize the sum over all projects of ('NPV (k$)'[i] * x[i]); that is, maximize total expected NPV of selected projects.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The sum over all projects of ('Capital (k$)'[i] * x[i]) ≤ 1,000.
    -   Constraint 2 (Mutually Exclusive Projects 4 & 7): x[4] + x[7] ≤ 1.
    -   Constraint 3 (Pre-requisite: Project 6 requires 1): x[6] ≤ x[1].
    -   Constraint 4 (Contingent: Project 10 requires 5): x[10] ≤ x[5].
    -   Constraint 5 (Binary domain): For all i, x[i] ∈ {0,1}.
[Abstract Model Plan END]