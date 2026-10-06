[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies: (a) a budget constraint, (b) a mutually exclusive constraint between Projects 4 and 7, (c) a pre-requisite constraint where Project 6 can only be selected if Project 1 is also selected, and (d) a contingent constraint where Project 10 can only be selected if Project 5 is also selected. All required data is in project.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (Project ID from 1 to 110).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column gives the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column gives the capital required for each project.
    -   Constraint RHS: Budget limit is 1,000 k$ (from the query, not the CSV). Logical dependencies are defined by Project IDs (from the query).
6.  **Formulate Objective:** Maximize the sum of selected projects' NPVs, i.e., maximize sum over all projects of `NPV[i] * x[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital invested in selected projects cannot exceed 1,000 k$. Formally, sum over all projects of `Capital[i] * x[i] <= 1,000`.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 or Project 7 can be selected: `x[4] + x[7] <= 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 can only be selected if Project 1 is also selected: `x[6] <= x[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 can only be selected if Project 5 is also selected: `x[10] <= x[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `x[i]` is binary (0 or 1).
[Abstract Model Plan END]