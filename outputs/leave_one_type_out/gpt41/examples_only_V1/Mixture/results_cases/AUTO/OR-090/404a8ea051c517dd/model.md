[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of production/purchase batches for each of 100 products (P1–P100), where each batch is 10 units, to maximize total profit. The solution must not exceed available capacities of three resources (R1, R2, R3), and for each product, total produced units cannot exceed its upper demand bound (unmet demand is allowed).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer variables for batch counts, linear constraints).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}), as listed in factory_products_100.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch_size_units (fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource usage per unit, from factory_products_100.csv), batch_size_units (to convert to per-batch usage).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv), and 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches x[i]) × (batch_size_units) × (profit_per_unit for product i).
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource (R1, R2, R3), the total consumption across all products (sum over i of x[i] × batch_size_units × r{j}_per_unit) must not exceed the corresponding resource's capacity from resources_capacities.csv.
    -   Constraint 2 (Demand Upper Bound): For each product i, the total produced units (x[i] × batch_size_units) must not exceed upper_demand_units[i] (from factory_products_100.csv).
    -   Constraint 3 (Batch Integrality): For each product i, x[i] must be an integer ≥ 0 (no fractional batches, no negative production).
[Abstract Model Plan END]