[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre to maximize the total value, subject to a total weight limit of 15 units. Each item can either be selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by $i$, where $i$ ranges over all rows in value.csv.
4.  **Define Decision Variables:**
    -   $x_i$ = 1 if item $i$ is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'value' column in value.csv (the value of each item).
    -   Constraint coefficients: 'weight' column in value.csv (the weight of each item).
    -   Constraint RHS: The total weight limit is 15 units (given in the query).
6.  **Formulate Objective:** Maximize the sum of the values of the selected items, i.e., maximize $\sum_{i} \text{value}_i \cdot x_i$.
7.  **Formulate Constraints:**
    -   Weight Limit: The total weight of selected items must not exceed 15 units, i.e., $\sum_{i} \text{weight}_i \cdot x_i \leq 15$.
    -   Binary Selection: For each item $i$, $x_i \in \{0,1\}$.
[Abstract Model Plan END]