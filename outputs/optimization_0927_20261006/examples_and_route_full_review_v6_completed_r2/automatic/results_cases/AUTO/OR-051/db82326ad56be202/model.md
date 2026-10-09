[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various coffee product units into multiple retail cabinets, maximizing the total value of products placed, while ensuring that the total weight in each cabinet does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-knapsack problem).
3.  **Define Index Sets:** The primary indices are:
    - Cabinets (from `capacity.csv`, indexed by `CabinetID`)
    - Products (from `products.csv`, indexed by `ProductName`)
4.  **Define Decision Variables:**
    - `x[i,j]` = Number of units of product `j` placed in cabinet `i`. Type: GRB.INTEGER (must be non-negative integers).
5.  **Identify Parameters (from Schema):**
    - Objective coefficients: `Value` (from `products.csv`, per product)
    - Constraint coefficients: `Weight` (from `products.csv`, per product)
    - Constraint RHS (limits): `Capacity` (from `capacity.csv`, per cabinet)
6.  **Formulate Objective:** Maximize the total value of all products placed in all cabinets, i.e., maximize the sum over all cabinets and products of (`Value` of product `j`) × (`x[i,j]`).
7.  **Formulate Constraints:**
    - Constraint 1 (Cabinet Capacity): For each cabinet `i`, the sum over all products `j` of (`Weight` of product `j`) × (`x[i,j]`) ≤ `Capacity` of cabinet `i`.
    - Constraint 2 (Non-negativity and Integrality): For all cabinets `i` and products `j`, `x[i,j]` ≥ 0 and integer.
[Abstract Model Plan END]