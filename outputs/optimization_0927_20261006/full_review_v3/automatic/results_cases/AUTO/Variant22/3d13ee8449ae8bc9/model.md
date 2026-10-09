[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each component family to maximize total value, subject to total weight and labor-hour limits, using data from option_catalog.csv and resource_limits.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a multi-choice (multiple-choice) knapsack problem.
3.  **Define Index Sets:** The primary indices are:
    - Component Families (from 'Family' in option_catalog.csv)
    - Options within each Family (from 'Option' in option_catalog.csv)
4.  **Define Decision Variables:**
    -   `x[c,o]` = 1 if option o is selected from family c; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column in option_catalog.csv (value of each option).
    -   Constraint coefficients: 'Weight' and 'LaborHours' columns in option_catalog.csv (resource usage per option).
    -   Constraint RHS (limits): 'Limit' column in resource_limits.csv, for resources 'Weight' and 'LaborHours'.
6.  **Formulate Objective:** Maximize the sum of 'Value' for all selected options, i.e., maximize total value across all chosen family-option pairs.
7.  **Formulate Constraints:**
    -   Constraint 1 (Exactly-One-Option-Per-Family): For each family c, the sum over all options o of x[c,o] = 1 (select exactly one option per family).
    -   Constraint 2 (Total Weight Limit): The sum over all families and options of 'Weight' * x[c,o] ≤ total weight limit from resource_limits.csv.
    -   Constraint 3 (Total Labor-Hour Limit): The sum over all families and options of 'LaborHours' * x[c,o] ≤ total labor-hour limit from resource_limits.csv.
    -   Constraint 4 (Binary Restrictions): All x[c,o] variables are binary (0 or 1).
[Abstract Model Plan END]