[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various products to multiple display shelves, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-knapsack allocation problem with integer variables).
3.  **Define Index Sets:** The primary indices are:
    - Shelves (indexed by i, from 'ShelfID' in capacity.csv)
    - Products (indexed by j, from 'ProductName' in products.csv)
4.  **Define Decision Variables:**
    - `x[i,j]` = Number of units of product j placed on shelf i. Type: GRB.INTEGER (must be integer and non-negative).
5.  **Identify Parameters (from Schema):**
    - Objective coefficients: 'Value' column from products.csv (value per unit of product j).
    - Constraint coefficients: 'Weight' column from products.csv (weight per unit of product j).
    - Constraint RHS: 'Capacity' column from capacity.csv (maximum total weight allowed on shelf i).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize sum over all shelves and products of (Value[j] * x[i,j]).
7.  **Formulate Constraints:**
    - Constraint 1 (Shelf Capacity): For each shelf i, the sum over all products j of (Weight[j] * x[i,j]) ≤ Capacity[i] (ensures total weight on each shelf does not exceed its capacity).
    - Constraint 2 (Non-negativity and Integrality): For all shelves i and products j, x[i,j] ≥ 0 and integer (cannot place negative or fractional units).
[Abstract Model Plan END]