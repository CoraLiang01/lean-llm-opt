[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre, maximizing the total value of selected items, subject to a total weight limit of 15 units. Each item can be either selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by `i`, where each `i` corresponds to a unique row in the CSV file.
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if item `i` is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'value' column (the value of each item).
    -   Constraint coefficients: 'weight' column (the weight of each item).
    -   Constraint RHS: The total weight limit is 15 (given in the query, not from the data).
6.  **Formulate Objective:** Maximize the sum over all items of (value[i] * x[i]), i.e., maximize the total value of selected items.
7.  **Formulate Constraints:**
    -   Constraint 1 (Weight Limit): The sum over all items of (weight[i] * x[i]) must be less than or equal to 15, ensuring the total weight of selected items does not exceed the counter's capacity.
    -   Constraint 2 (Binary Selection): For each item `i`, x[i] ∈ {0, 1}, enforcing that each item is either selected or not.
[Abstract Model Plan END]