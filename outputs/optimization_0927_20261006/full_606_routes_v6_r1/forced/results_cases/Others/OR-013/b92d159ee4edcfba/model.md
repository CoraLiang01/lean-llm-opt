[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal fulfillment quantities for each “4U” product in order to maximize total revenue, given deterministic demand, initial inventory, and no restocking during the sales period. The fulfillment for each product cannot exceed either its available inventory or its demand.
2.  **Identify Model Type:** Based on the query, this is a Linear Integer Programming (IP) problem (bounded knapsack with per-item upper limits).
3.  **Define Index Sets:** The primary index is the set of “4U” products (i ∈ Products), where Products is the subset of rows in the CSV whose 'Product Name' contains the substring "4U".
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of “4U” product i to fulfill (i.e., to sell). Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Revenue' column (per-unit revenue for each product i).
    -   Constraint coefficients and RHS:
        -   'Demand' column (maximum possible fulfillment for each product i).
        -   'Initial Inventory' column (available stock for each product i).
6.  **Formulate Objective:** Maximize total revenue from “4U” products, i.e., maximize the sum over all selected products of (schema['Revenue'][i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Limit): For each product i, x[i] ≤ schema['Initial Inventory'][i] (cannot fulfill more than available inventory).
    -   Constraint 2 (Demand Limit): For each product i, x[i] ≤ schema['Demand'][i] (cannot fulfill more than realized demand).
    -   Constraint 3 (Non-negativity and Integrality): For each product i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]