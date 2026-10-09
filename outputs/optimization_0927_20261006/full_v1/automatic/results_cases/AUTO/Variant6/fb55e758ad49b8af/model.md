[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each product family to form a promotional bundle, maximizing total value, while ensuring that the total shelf weight and budget usage of the selected options do not exceed specified limits. Each option's value, weight, and budget usage are given, as are the overall resource limits.
2.  **Identify Model Type:** Based on the query, this is a Binary Multi-Choice Knapsack Problem (a special case of Mixed-Integer Programming with binary variables).
3.  **Define Index Sets:** The primary indices are:
    - Families (f ∈ Families, from 'Family' column in option_catalog.csv)
    - Options within each family (o ∈ Options_f, from 'Option' column in option_catalog.csv)
4.  **Define Decision Variables:**
    -   `x[f,o]` = 1 if option o is selected for family f, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column in option_catalog.csv (value of each option).
    -   Constraint coefficients: 'Weight' and 'BudgetUse' columns in option_catalog.csv (resource usage per option).
    -   Constraint RHS (limits): 'Limit' column in resource_limits.csv, for resources 'Weight' and 'BudgetUse'.
6.  **Formulate Objective:** Maximize the total value of the bundle, i.e., maximize the sum over all families and options of (Value[f,o] * x[f,o]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Multi-choice selection): For each family f, sum over all options o in family f of x[f,o] = 1 (exactly one option selected per family).
    -   Constraint 2 (Shelf weight limit): Sum over all families and options of (Weight[f,o] * x[f,o]) ≤ Weight limit from resource_limits.csv.
    -   Constraint 3 (Budget usage limit): Sum over all families and options of (BudgetUse[f,o] * x[f,o]) ≤ BudgetUse limit from resource_limits.csv.
    -   Constraint 4 (Binary restrictions): x[f,o] ∈ {0,1} for all families f and options o.
[Abstract Model Plan END]