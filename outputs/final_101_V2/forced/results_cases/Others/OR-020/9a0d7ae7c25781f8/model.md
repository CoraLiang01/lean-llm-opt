[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for each product, how many units to fulfill (sell) in order to maximize total revenue, given deterministic demand and limited initial inventory for each product.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem (all variables are continuous or integer, no binary activation or fixed-charge costs).
3.  **Define Index Sets:** The primary index is Products (i), where each product corresponds to a row in the CSV file.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of product i to fulfill (sell). Type: GRB.INTEGER (since demand and inventory are integer-valued and sales are in units).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Revenue' column (per-unit revenue for each product).
    -   Constraint coefficients: Not needed (each constraint applies directly to a single product).
    -   Constraint RHS (limits): 'Demand' column (maximum possible sales per product), 'Initial Inventory' column (maximum available stock per product).
6.  **Formulate Objective:** Maximize total revenue, i.e., maximize sum over all products of (schema['Revenue'][i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand fulfillment): For each product i, x[i] ≤ schema['Demand'][i] (cannot fulfill more than demand).
    -   Constraint 2 (Inventory limit): For each product i, x[i] ≤ schema['Initial Inventory'][i] (cannot fulfill more than available inventory).
    -   Constraint 3 (Non-negativity): For each product i, x[i] ≥ 0 (cannot fulfill negative units).
    -   Constraint 4 (Integrality): For each product i, x[i] is integer (since units are discrete).
[Abstract Model Plan END]