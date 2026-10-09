[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre to maximize the total value, subject to a total weight limit of 15 units. Each item can either be selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by $i$, where each $i$ corresponds to a row in value.csv.
4.  **Define Decision Variables:**
    -   $x[i]$ = 1 if item $i$ is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (item value) will come from column: 'value'.
    -   Constraint coefficients (item weight) will come from column: 'weight'.
    -   Constraint RHS (total weight limit) is 15 (from the query).
6.  **Formulate Objective:** Maximize the total value of selected items: $\sum_{i} \text{value}[i] \cdot x[i]$.
7.  **Formulate Constraints:**
    -   Constraint 1 (Weight Limit): $\sum_{i} \text{weight}[i] \cdot x[i] \leq 15$.
    -   Constraint 2 (Binary Selection): $x[i] \in \{0,1\}$ for all items $i$.
[Abstract Model Plan END]