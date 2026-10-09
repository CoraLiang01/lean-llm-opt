[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre to maximize the total value, subject to a total weight limit of 15 units. Each item can either be selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by $i$, where $i$ ranges over all rows in value.csv.
4.  **Define Decision Variables:**
    -   $x_i$ = 1 if item $i$ is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'value' column in value.csv (the value of each item).
    -   Constraint coefficients: 'weight' column in value.csv (the weight of each item).
    -   Constraint RHS: The total weight limit is 15 (given in the query, not from the file).
6.  **Formulate Objective:** Maximize the sum over all items of (value[i] * x[i]); that is, maximize the total value of selected items.
7.  **Formulate Constraints:**
    -   Constraint 1 (Weight Limit): The sum over all items of (weight[i] * x[i]) must be less than or equal to 15.
    -   Constraint 2 (Binary Selection): For each item $i$, $x_i$ is binary (0 or 1).
[Abstract Model Plan END]