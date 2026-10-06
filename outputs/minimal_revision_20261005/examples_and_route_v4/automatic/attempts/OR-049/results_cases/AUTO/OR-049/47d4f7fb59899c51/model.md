[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of multiple product types to multiple display shelves, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-dimensional integer knapsack allocation).
3.  **Define Index Sets:** The primary indices are:
    - Shelves (from `capacity.csv`, indexed by `ShelfID`)
    - Products (from `products.csv`, indexed by `ProductName`)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product j placed on shelf i. Type: GRB.INTEGER (must be integer and non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from `products.csv` (the value per unit of each product).
    -   Constraint coefficients: 'Weight' column from `products.csv` (the weight per unit of each product).
    -   Constraint RHS (limits): 'Capacity' column from `capacity.csv` (the maximum total weight allowed on each shelf).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize the sum over all shelves and products of (Value of product j) × (number of units of product j on shelf i):  
    Maximize ∑₍i∈Shelves₎ ∑₍j∈Products₎ Value[j] × x[i,j]
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf i, the total weight of all products placed on that shelf cannot exceed its capacity:  
        ∑₍j∈Products₎ Weight[j] × x[i,j] ≤ Capacity[i]  for all shelves i
    -   Constraint 2 (Non-negativity and Integrality):  
        x[i,j] ≥ 0 and integer, for all shelves i and products j
    -   (No explicit upper bound on x[i,j] unless further product or shelf-specific limits are provided in the data or query.)
[Abstract Model Plan END]