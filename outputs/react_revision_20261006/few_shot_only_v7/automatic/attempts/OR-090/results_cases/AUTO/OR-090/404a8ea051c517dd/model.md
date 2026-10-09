[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products (P1–P100) in a single planning horizon, where each batch is 10 units, to maximize total profit. Production is limited by three resource capacities and by each product’s upper demand (in units), but it is allowed to under-produce (unmet demand is permitted). All resource and demand limits must be respected, and only whole batches can be produced.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary indices are Products (i ∈ {P1, ..., P100}) and Resources (r ∈ {R1, R2, R3}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches produced) × (batch size) × (profit per unit), i.e., maximize sum_i [ x[i] * batch_size_units * profit_per_unit[i] ].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumption across all products cannot exceed the available capacity. For each r:
        - sum_i [ x[i] * batch_size_units * r{r}_per_unit[i] ] ≤ capacity[r]
        - (where r{r}_per_unit[i] is the per-unit consumption for resource r for product i)
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed its upper demand:
        - x[i] * batch_size_units ≤ upper_demand_units[i]
        - (since only whole batches can be produced, this may result in not fully meeting demand if upper_demand_units[i] is not a multiple of batch_size_units)
    -   Constraint 3 (Non-negativity and Integrality): For all i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]