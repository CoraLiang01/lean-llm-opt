[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to maximize total revenue by selecting how many units to fulfill for each product classified as ‘Books’, subject to inventory and demand constraints. Only products with 'Books' in the 'Product_Name' are considered. The decision variable for each such product is the number of units fulfilled, which cannot exceed either the initial inventory or the demand for that product.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary index is the set of products i where 'Product_Name' starts with 'Books' (i.e., all rows in the CSV where 'Product_Name' matches this prefix).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of 'Books' product i to fulfill. Type: GRB.CONTINUOUS (or GRB.INTEGER if only whole units are allowed; the schema uses int64 for demand, but the variable type is not explicitly specified in the query).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Revenue' column (per-unit revenue for each product i).
    -   Constraint coefficients: Each variable is directly bounded by 'Demand' and 'Initial Inventory' columns for product i.
    -   Constraint RHS: 'Demand' and 'Initial Inventory' columns (upper bounds for each x[i]).
6.  **Formulate Objective:** Maximize the sum over all selected 'Books' products of (schema['Revenue'][i] * x[i]); that is, maximize total revenue from fulfilled units of 'Books' products.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Limit): For each 'Books' product i, x[i] ≤ schema['Demand'][i] (cannot fulfill more than demand).
    -   Constraint 2 (Inventory Limit): For each 'Books' product i, x[i] ≤ schema['Initial Inventory'][i] (cannot fulfill more than available inventory).
    -   Constraint 3 (Non-negativity): For each 'Books' product i, x[i] ≥ 0.
[Abstract Model Plan END]