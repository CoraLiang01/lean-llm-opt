[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various products to multiple display shelves, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-dimensional integer knapsack allocation).
3.  **Define Index Sets:** The primary indices are:
    - Shelves (from `capacity.csv`, indexed by `ShelfID`)
    - Products (from `products.csv`, indexed by `ProductName`)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product `j` placed on shelf `i`. Type: GRB.INTEGER (must be integer and non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from `products.csv` (the value per unit of each product).
    -   Constraint coefficients: 'Weight' column from `products.csv` (the weight per unit of each product).
    -   Constraint RHS (limits): 'Capacity' column from `capacity.csv` (the maximum total weight allowed on each shelf).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize the sum over all shelves and products of (product value × number of units placed):  
    Maximize: sum over all shelves i and products j of `Value[j] * x[i,j]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf `i`, the total weight of all products placed on that shelf cannot exceed its capacity:  
        sum over all products j of `Weight[j] * x[i,j]` ≤ `Capacity[i]` for each shelf i.
    -   Constraint 2 (Non-negativity and Integrality): For all shelves i and products j, `x[i,j]` ≥ 0 and integer.
    -   (No explicit upper bound on product units per shelf is given; if there are additional limits, they would be included as further constraints.)
[Abstract Model Plan END]