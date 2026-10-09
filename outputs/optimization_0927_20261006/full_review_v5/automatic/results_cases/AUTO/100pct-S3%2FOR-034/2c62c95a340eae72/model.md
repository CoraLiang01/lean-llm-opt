[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre to maximize total value, subject to a total weight limit of 15 units. Each item can either be selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by $i$, where $i$ ranges over all rows in value.csv.
4.  **Define Decision Variables:**
    -   $x_i$ = 1 if item $i$ is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (item value) will come from column: 'value'.
    -   Constraint coefficients (item weight) will come from column: 'weight'.
    -   Constraint RHS (total weight limit) is the fixed value 15 (from the query).
6.  **Formulate Objective:** Maximize the total value of selected items: $\sum_{i} \text{value}[i] \cdot x_i$.
7.  **Formulate Constraints:**
    -   Constraint 1 (Weight Limit): $\sum_{i} \text{weight}[i] \cdot x_i \leq 15$ (total weight of selected items does not exceed the counter's capacity).
    -   Constraint 2 (Binary Selection): $x_i \in \{0,1\}$ for all items $i$ (each item is either selected or not).
[Abstract Model Plan END]