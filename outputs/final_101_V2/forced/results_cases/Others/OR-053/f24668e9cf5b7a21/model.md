[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various product units to different display shelves in a store, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-dimensional integer knapsack allocation).
3.  **Define Index Sets:** The primary indices are:
    - Shelves (from `capacity.csv`, indexed by `ShelfID`, 1 to 10)
    - Products (from `products.csv`, indexed by `ProductName`, 1 to 20)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product j placed on shelf i. Type: GRB.INTEGER (must be non-negative integers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from `products.csv` (the value per unit of each product).
    -   Constraint coefficients: 'Weight' column from `products.csv` (the weight per unit of each product).
    -   Constraint RHS (limits): 'Capacity' column from `capacity.csv` (the maximum total weight allowed on each shelf).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize the sum over all shelves and products of (Value of product j) × (number of units of product j on shelf i):  
    Maximize ∑₍i∈Shelves₎ ∑₍j∈Products₎ Value[j] × x[i,j]
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf i, the total weight of all products placed on that shelf cannot exceed its capacity:  
        For all i: ∑₍j∈Products₎ Weight[j] × x[i,j] ≤ Capacity[i]
    -   Constraint 2 (Non-negativity and Integrality): For all shelves i and products j:  
        x[i,j] ≥ 0 and integer
    -   (No explicit upper bound on x[i,j] unless further product availability or shelf-specific limits are provided in the data.)
[Abstract Model Plan END]