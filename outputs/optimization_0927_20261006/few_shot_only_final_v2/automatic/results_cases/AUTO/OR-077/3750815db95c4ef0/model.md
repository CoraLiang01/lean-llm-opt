[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre to maximize the total value, subject to a total weight limit of 15 units. Each item can either be selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is Items (i ∈ Items), where each item is uniquely identified by the 'item' column in value.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if item i is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'value' column in value.csv (the value of each item).
    -   Constraint coefficients: 'weight' column in value.csv (the weight of each item).
    -   Constraint RHS: The total weight limit is 15 units (given in the query).
6.  **Formulate Objective:** Maximize the total value of selected items, i.e., maximize sum over i of value[i] * x[i].
7.  **Formulate Constraints:**
    -   Constraint 1 (Weight Limit): The sum over i of weight[i] * x[i] ≤ 15 (total weight of selected items does not exceed the counter limit).
    -   Constraint 2 (Binary Selection): For all i, x[i] ∈ {0, 1} (each item is either selected or not).
[Abstract Model Plan END]