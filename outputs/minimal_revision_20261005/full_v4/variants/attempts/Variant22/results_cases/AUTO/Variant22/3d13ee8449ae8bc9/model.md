[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each component family to maximize total value, subject to total weight and labor-hour limits. Each option has a value, weight, and labor-hour requirement. The selection is modeled with binary variables for each family-option pair.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a multi-choice (multiple-choice) knapsack problem.
3.  **Define Index Sets:** The primary indices are:
    - Component Families (from 'Family' column in option_catalog.csv)
    - Options within each Family (from 'Option' column in option_catalog.csv)
4.  **Define Decision Variables:**
    -   `x[c,o]` = 1 if option o is selected from family c; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column in option_catalog.csv (value of each option).
    -   Constraint coefficients: 'Weight' and 'LaborHours' columns in option_catalog.csv (resource usage per option).
    -   Constraint RHS (limits): 'Limit' column in resource_limits.csv, for resources 'Weight' and 'LaborHours'.
6.  **Formulate Objective:** Maximize the sum of the 'Value' of all selected options, i.e., maximize total value across all families by summing Value[c,o] * x[c,o] for all family-option pairs.
7.  **Formulate Constraints:**
    -   Constraint 1 (Exactly-One-Option-Per-Family): For each family c, the sum over all options o of x[c,o] = 1. (Exactly one option must be selected from each family.)
    -   Constraint 2 (Total Weight Limit): The sum over all family-option pairs of Weight[c,o] * x[c,o] ≤ total weight limit (from resource_limits.csv, 'Weight').
    -   Constraint 3 (Total Labor-Hour Limit): The sum over all family-option pairs of LaborHours[c,o] * x[c,o] ≤ total labor-hour limit (from resource_limits.csv, 'LaborHours').
    -   Constraint 4 (Binary Restrictions): All x[c,o] variables are binary (0 or 1).
[Abstract Model Plan END]