[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each component family, maximizing total value, while ensuring that the total weight and total labor-hours of the selected options do not exceed specified resource limits. Each option has associated value, weight, and labor-hour requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a multi-choice (multiple-choice) knapsack problem with binary selection variables.
3.  **Define Index Sets:** The primary indices are:
    - Component Families (from 'Family' column in option_catalog.csv)
    - Options within each Family (from 'Option' column in option_catalog.csv)
    - Resources (from 'Resource' column in resource_limits.csv; specifically 'Weight' and 'LaborHours')
4.  **Define Decision Variables:**
    -   `x[c,o]` = 1 if option o is selected from family c; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column in option_catalog.csv (value of each option).
    -   Constraint coefficients: 'Weight' and 'LaborHours' columns in option_catalog.csv (resource usage per option).
    -   Constraint RHS (limits): 'Limit' column in resource_limits.csv (total allowed for each resource).
6.  **Formulate Objective:** Maximize the sum of the 'Value' of all selected options, i.e., maximize total value across all families by summing 'Value[c,o] * x[c,o]' over all family-option pairs.
7.  **Formulate Constraints:**
    -   Constraint 1 (Exactly-One-Option-Per-Family): For each family c, the sum over all options o of x[c,o] = 1. (Exactly one option must be selected from each family.)
    -   Constraint 2 (Weight Limit): The sum over all family-option pairs of 'Weight[c,o] * x[c,o]' ≤ the 'Weight' limit from resource_limits.csv.
    -   Constraint 3 (Labor-Hours Limit): The sum over all family-option pairs of 'LaborHours[c,o] * x[c,o]' ≤ the 'LaborHours' limit from resource_limits.csv.
    -   Constraint 4 (Binary Restrictions): All x[c,o] variables are binary (0 or 1).
[Abstract Model Plan END]