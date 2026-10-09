[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each product family to form a promotional bundle, maximizing total value, while ensuring that the total shelf weight and budget usage of the selected options do not exceed specified limits. Each option has associated value, weight, and budget usage.
2.  **Identify Model Type:** Based on the query, this is a Binary Multi-Choice Knapsack Problem (a special case of Mixed-Integer Programming with binary variables).
3.  **Define Index Sets:** The primary indices are:
    - Families (f), as given by the 'Family' column in option_catalog.csv.
    - Options (o), as given by the 'Option' column in option_catalog.csv, with each option belonging to a family.
4.  **Define Decision Variables:**
    -   `x[f,o]` = 1 if option o is selected for family f, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from option_catalog.csv (value of each option).
    -   Constraint coefficients: 'Weight' and 'BudgetUse' columns from option_catalog.csv (resource usage per option).
    -   Constraint RHS (limits): 'Limit' column from resource_limits.csv, for resources 'Weight' and 'BudgetUse'.
6.  **Formulate Objective:** Maximize the total value of the bundle, i.e., maximize the sum over all families and options of (Value[f,o] * x[f,o]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Family Selection): For each family f, sum over all options o in family f of x[f,o] = 1 (exactly one option selected per family).
    -   Constraint 2 (Shelf Weight Limit): The sum over all families and options of (Weight[f,o] * x[f,o]) ≤ Weight limit from resource_limits.csv.
    -   Constraint 3 (Budget Usage Limit): The sum over all families and options of (BudgetUse[f,o] * x[f,o]) ≤ BudgetUse limit from resource_limits.csv.
    -   Constraint 4 (Binary Restriction): All x[f,o] are binary variables (0 or 1).
[Abstract Model Plan END]