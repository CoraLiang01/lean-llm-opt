[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various product units to different display shelves in a store, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a multi-knapsack allocation problem.
3.  **Define Index Sets:** The primary indices are:
    - Shelves (indexed by i, from capacity.csv, ShelfID 1–10)
    - Products (indexed by j, from products.csv, ProductName 1–20)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product j to place on shelf i. Type: GRB.INTEGER (must be integer and non-negative).
5.  **Identify Parameters (from Schema):**
    -   Value of each product: from products.csv, column 'Value' (indexed by j).
    -   Weight of each product: from products.csv, column 'Weight' (indexed by j).
    -   Shelf capacity: from capacity.csv, column 'Capacity' (indexed by i).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize sum over all shelves and products of (Value[j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf i, the sum over all products j of (Weight[j] * x[i,j]) ≤ Capacity[i]. This ensures the total weight on each shelf does not exceed its capacity.
    -   Constraint 2 (Non-negativity and Integrality): For all shelves i and products j, x[i,j] ≥ 0 and integer.
    -   (If there are additional business rules, such as product availability limits or shelf-specific restrictions, these would be added as further constraints, but none are specified in the query or schema.)
[Abstract Model Plan END]