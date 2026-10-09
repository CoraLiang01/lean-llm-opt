[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of various products to different display shelves in a store, maximizing the total value of displayed products, while ensuring that the total weight of products on each shelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-dimensional integer knapsack allocation).
3.  **Define Index Sets:** The primary indices are Shelves (from `capacity.csv`, indexed by `ShelfID`) and Products (from `products.csv`, indexed by `ProductName`).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of product `j` placed on shelf `i`. Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `Value` (from `products.csv`, column 'Value' for each product).
    -   Constraint coefficients: `Weight` (from `products.csv`, column 'Weight' for each product).
    -   Constraint RHS (limits): `Capacity` (from `capacity.csv`, column 'Capacity' for each shelf).
6.  **Formulate Objective:** Maximize the total value of products placed on all shelves, i.e., maximize the sum over all shelves and products of (`Value` of product `j`) × (`x[i,j]`).
7.  **Formulate Constraints:**
    -   Shelf Capacity Constraint: For each shelf `i`, the sum over all products `j` of (`Weight` of product `j`) × (`x[i,j]`) ≤ (`Capacity` of shelf `i`).
    -   Non-negativity and Integrality: For all shelves `i` and products `j`, `x[i,j]` ≥ 0 and integer.
[Abstract Model Plan END]