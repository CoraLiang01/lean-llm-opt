[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select exactly one option from each product family to form a promotional bundle, maximizing total value, while ensuring that the total shelf weight and budget usage of the selected options do not exceed specified limits. Each option has associated value, weight, and budget usage.
2.  **Identify Model Type:** Based on the query, this is a Binary Multi-Choice Knapsack Problem (a special case of Mixed Integer Programming, MIP).
3.  **Define Index Sets:** The primary indices are:
    - Product Families (from 'Family' column in option_catalog.csv)
    - Options within each Family (from 'Option' column in option_catalog.csv)
4.  **Define Decision Variables:**
    -   `x[f,o]` = 1 if option o is selected for family f, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column in option_catalog.csv (value of each option).
    -   Constraint coefficients:
        - 'Weight' column in option_catalog.csv (shelf weight of each option).
        - 'BudgetUse' column in option_catalog.csv (budget usage of each option).
    -   Constraint RHS (limits): 'Limit' column in resource_limits.csv, for resources 'Weight' and 'BudgetUse'.
6.  **Formulate Objective:** Maximize the total value of the selected bundle, i.e., maximize the sum of 'Value' for all selected options (sum over all families and their options: sum(Value[f,o] * x[f,o])).
7.  **Formulate Constraints:**
    -   Constraint 1 (Family Selection): For each family f, exactly one option must be selected: sum over o of x[f,o] = 1 for all f.
    -   Constraint 2 (Shelf Weight Limit): The total shelf weight of selected options must not exceed the limit: sum over all f,o of Weight[f,o] * x[f,o] ≤ resource_limits['Weight'].
    -   Constraint 3 (Budget Usage Limit): The total budget usage of selected options must not exceed the limit: sum over all f,o of BudgetUse[f,o] * x[f,o] ≤ resource_limits['BudgetUse'].
    -   Constraint 4 (Binary Restriction): All x[f,o] are binary variables (0 or 1).
[Abstract Model Plan END]