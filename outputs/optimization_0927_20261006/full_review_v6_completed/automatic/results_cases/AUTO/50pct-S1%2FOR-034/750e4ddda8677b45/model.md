[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of items for display in a shopping centre, maximizing the total value of selected items without exceeding a total weight limit of 15 units. Each item can be either selected or not (binary decision).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a 0-1 Knapsack problem.
3.  **Define Index Sets:** The primary index is the set of items, indexed by $i$, where $i$ ranges over all rows in value.csv.
4.  **Define Decision Variables:**
    -   $x_i$ = 1 if item $i$ is selected for display, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'value' column provides the value of each item ($\text{value}_i$).
    -   Constraint coefficients: 'weight' column provides the weight of each item ($\text{weight}_i$).
    -   Constraint RHS: The total weight limit is 15 units (given in the query).
6.  **Formulate Objective:** Maximize the total value of selected items: $\sum_{i} \text{value}_i \cdot x_i$.
7.  **Formulate Constraints:**
    -   Weight Limit: $\sum_{i} \text{weight}_i \cdot x_i \leq 15$ (the total weight of selected items cannot exceed 15 units).
    -   Binary Selection: $x_i \in \{0,1\}$ for all items $i$ (each item is either selected or not).
[Abstract Model Plan END]