[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to maximize total revenue by fulfilling demand for products classified under ‘Baby’, using only the available initial inventory for each such product. The number of units fulfilled for each product cannot exceed either its demand or its initial inventory.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary index is the set of products where ‘Product Name’ contains the substring ‘Baby’ (i.e., all ‘Baby’ products in the data).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of ‘Baby’ product i to fulfill. Type: GRB.INTEGER (since units are discrete).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: ‘Revenue’ column (per-unit revenue for each product i).
    -   Constraint coefficients and RHS:
        -   ‘Demand’ column (maximum demand for each product i).
        -   ‘Initial Inventory’ column (available inventory for each product i).
6.  **Formulate Objective:** Maximize the sum over all ‘Baby’ products i of (schema['Revenue'][i] * x[i]); that is, maximize total revenue from fulfilled units.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Fulfillment): For each ‘Baby’ product i, x[i] ≤ schema['Demand'][i] (cannot fulfill more than demand).
    -   Constraint 2 (Inventory Limit): For each ‘Baby’ product i, x[i] ≤ schema['Initial Inventory'][i] (cannot fulfill more than available inventory).
    -   Constraint 3 (Non-negativity and Integrality): For each ‘Baby’ product i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]