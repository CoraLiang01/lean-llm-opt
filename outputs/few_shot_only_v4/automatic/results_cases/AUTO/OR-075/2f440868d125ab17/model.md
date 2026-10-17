[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of 110 potential investment projects to maximize total expected Net Present Value (NPV), subject to a total capital budget and several strategic dependency constraints (mutual exclusivity, pre-requisite, and contingent selection).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a binary project selection/knapsack problem with logical constraints).
3.  **Define Index Sets:** The primary index is the set of Projects, indexed by `i` (Project ID from 1 to 110).
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if project `i` is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'NPV (k$)' column from project.csv, mapped as `npv[i]`.
    -   Constraint coefficients: 'Capital (k$)' column from project.csv, mapped as `capital[i]`.
    -   Constraint RHS: Budget limit is 1,000 (k$); logical constraints reference specific Project IDs (4, 7, 6, 1, 10, 5).
6.  **Formulate Objective:** Maximize the sum over all projects of their expected NPV times the selection variable: maximize `sum_i npv[i] * y[i]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Budget Constraint): The total capital invested in selected projects cannot exceed 1,000 k$: `sum_i capital[i] * y[i] <= 1000`.
    -   Constraint 2 (Mutually Exclusive Constraint): At most one of Project 4 or Project 7 can be selected: `y[4] + y[7] <= 1`.
    -   Constraint 3 (Pre-requisite Constraint): Project 6 can only be selected if Project 1 is selected: `y[6] <= y[1]`.
    -   Constraint 4 (Contingent Constraint): Project 10 can only be selected if Project 5 is selected: `y[10] <= y[5]`.
    -   Constraint 5 (Binary Domain): For all projects, `y[i]` is binary (0 or 1).
[Abstract Model Plan END]