[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various product units to different display shelves in a store, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a multi-dimensional integer knapsack allocation problem.
3.  **Define Index Sets:** The primary indices are:
    - Shelves (indexed by i, from the 'ShelfID' column in capacity.csv; 10 shelves)
    - Products (indexed by j, from the 'ProductName' column in products.csv; 20 products)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product j to be placed on shelf i. Type: GRB.INTEGER (must be non-negative integers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (value per unit of each product).
    -   Constraint coefficients: 'Weight' column from products.csv (weight per unit of each product).
    -   Constraint RHS (limits): 'Capacity' column from capacity.csv (maximum total weight per shelf).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize sum over all shelves and products of (Value[j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf i, the sum over all products j of (Weight[j] * x[i,j]) ≤ Capacity[i]. This ensures the total weight on each shelf does not exceed its capacity.
    -   Constraint 2 (Non-negativity and Integrality): For all shelves i and products j, x[i,j] ≥ 0 and integer.
    -   (No explicit upper bound on product units per shelf is given, so only shelf capacity restricts allocations.)
[Abstract Model Plan END]