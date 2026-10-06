[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre, maximizing the total value of selected items without exceeding a total weight limit of 15 units. Each item can either be selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by `i`, where each `i` corresponds to a row in the CSV (item 1 to item 140).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if item `i` is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'value' column (the value of each item).
    -   Constraint coefficients: 'weight' column (the weight of each item).
    -   Constraint RHS: The total weight limit is 15 (given in the query, not from the CSV).
6.  **Formulate Objective:** Maximize the sum of the values of selected items, i.e., maximize sum over all items of (value[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Weight Limit): The total weight of selected items must not exceed 15, i.e., sum over all items of (weight[i] * x[i]) ≤ 15.
    -   Constraint 2 (Binary Selection): For each item i, x[i] ∈ {0, 1}.
[Abstract Model Plan END]