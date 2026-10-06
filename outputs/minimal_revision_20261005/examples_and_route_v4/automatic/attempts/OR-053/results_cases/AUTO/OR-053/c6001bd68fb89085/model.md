[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various product units to different display shelves in a store, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-dimensional multiple knapsack problem).
3.  **Define Index Sets:** The primary indices are:
    - Shelves (indexed by i, from capacity.csv, ShelfID 1–10)
    - Products (indexed by j, from products.csv, ProductName 1–20)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product j to place on shelf i. Type: GRB.INTEGER (must be non-negative integers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (value per unit of product j).
    -   Constraint coefficients: 'Weight' column from products.csv (weight per unit of product j).
    -   Constraint RHS (limits): 'Capacity' column from capacity.csv (maximum total weight per shelf i).
6.  **Formulate Objective:** Maximize the total value of all products placed on all shelves, i.e., maximize the sum over all shelves and products of (Value of product j) × (number of units of product j on shelf i):  
    Maximize ∑₍ᵢ₎ ∑₍ⱼ₎ Value[j] × x[i,j]
7.  **Formulate Constraints:**
    -   Constraint 1 (Shelf Capacity): For each shelf i, the total weight of all products placed on that shelf cannot exceed its capacity:  
        ∑₍ⱼ₎ Weight[j] × x[i,j] ≤ Capacity[i]  for all shelves i
    -   Constraint 2 (Non-negativity and Integrality):  
        x[i,j] ≥ 0 and integer, for all shelves i and products j
    -   (No explicit upper bound on x[i,j] unless further product availability or shelf-specific restrictions are provided in the data or query.)
[Abstract Model Plan END]