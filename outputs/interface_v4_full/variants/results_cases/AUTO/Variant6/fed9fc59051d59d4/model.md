[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each product family to form a promotional bundle, maximizing total value, while ensuring that the total shelf weight and budget usage of the selected options do not exceed specified limits. Each option has associated value, weight, and budget usage.
2.  **Identify Model Type:** Based on the query, this is a Binary Multi-Choice Knapsack Problem (a special case of Mixed Integer Programming, MIP).
3.  **Define Index Sets:** The primary indices are:
    - Product Families (from 'Family' column in option_catalog.csv)
    - Options within each Family (from 'Option' column in option_catalog.csv)
4.  **Define Decision Variables:**
    -   `x[family, option]` = 1 if option is selected for family, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column in option_catalog.csv (value of each option).
    -   Constraint coefficients:
        - 'Weight' column in option_catalog.csv (shelf weight per option).
        - 'BudgetUse' column in option_catalog.csv (budget usage per option).
    -   Constraint RHS (limits): 'Limit' column in resource_limits.csv, for resources 'Weight' and 'BudgetUse'.
6.  **Formulate Objective:** Maximize the total value of the bundle, i.e., sum over all families and options of (Value * x[family, option]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Family Selection): For each family, select exactly one option: sum over options in each family of x[family, option] = 1.
    -   Constraint 2 (Shelf Weight Limit): The total shelf weight of selected options does not exceed the limit: sum over all families and options of (Weight * x[family, option]) ≤ Weight limit from resource_limits.csv.
    -   Constraint 3 (Budget Limit): The total budget usage of selected options does not exceed the limit: sum over all families and options of (BudgetUse * x[family, option]) ≤ BudgetUse limit from resource_limits.csv.
    -   Constraint 4 (Binary Restriction): All x[family, option] variables are binary (0 or 1).
[Abstract Model Plan END]