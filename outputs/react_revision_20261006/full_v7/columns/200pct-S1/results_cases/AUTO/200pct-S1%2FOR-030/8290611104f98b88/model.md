[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic project dependencies (mutual exclusivity, prerequisites, and contingent selections).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (where `i` ranges over all 110 projects in project.csv, identified by 'Project ID').
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column gives the expected NPV for each project.
    -   Constraint coefficients: 'Capital (k$)' column gives the capital required for each project.
    -   Constraint RHS: Budget limit is 1,000 k$ (from the query, not the CSV).
    -   Project dependencies: Specific project IDs and names are mapped as follows (from preview and query):
        - Project 4: R&D Initiative Alpha
        - Project 7: Global Expansion Pilot
        - Project 6: System Automation
        - Project 1: Infrastructure Upgrade
        - Project 10: Customer Experience Platform
        - Project 5: Staff Training Program
6.  **Formulate Objective:** Maximize the sum of selected projects' NPVs: maximize sum over all projects of `NPV[i] * y[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital of selected projects cannot exceed 1,000 k$: sum over all projects of `Capital[i] * y[i] <= 1000`.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 or Project 7 can be selected: `y[4] + y[7] <= 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 can only be selected if Project 1 is selected: `y[6] <= y[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 can only be selected if Project 5 is selected: `y[10] <= y[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `y[i]` ∈ {0,1}.
[Abstract Model Plan END]