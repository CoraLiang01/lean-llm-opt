[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various products to multiple display shelves, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-dimensional integer knapsack allocation).
3.  **Define Index Sets:** The primary indices are Shelves (from capacity.csv, indexed by ShelfID) and Products (from products.csv, indexed by ProductName).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product j placed on shelf i. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (value per unit of each product).
    -   Constraint coefficients: 'Weight' column from products.csv (weight per unit of each product).
    -   Constraint RHS (limits): 'Capacity' column from capacity.csv (maximum total weight per shelf).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize the sum over all shelves and products of (Value of product j) × (number of units of product j on shelf i).
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf i, the sum over all products j of (Weight of product j) × (number of units of product j on shelf i) must be less than or equal to the shelf's Capacity.
    -   Constraint 2 (Non-negativity and Integrality): For all shelves i and products j, x[i,j] ≥ 0 and integer.
[Abstract Model Plan END]