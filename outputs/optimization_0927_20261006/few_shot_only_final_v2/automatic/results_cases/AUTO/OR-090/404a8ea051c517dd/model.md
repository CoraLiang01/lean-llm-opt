[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products, where each batch is 10 units, in order to maximize total profit. The plan must respect per-product upper demand limits (in units, not multiples of 10) and total available capacities of three shared resources (R1, R2, R3). Unmet demand is allowed, but overproduction is not.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with resource constraints and batch-size integer variables.
3.  **Define Index Sets:** The primary indices are Products (i ∈ {P1, ..., P100}) and Resources (r ∈ {R1, R2, R3}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size 'batch_size_units' (fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS: 'upper_demand_units' (maximum allowed units per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (batch_size_units * x[i] * profit_per_unit[i]), i.e., maximize sum_i (10 * x[i] * profit_per_unit[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumption across all products cannot exceed its capacity: sum_i (batch_size_units * x[i] * r_per_unit[i][r]) ≤ capacity[r].
    -   Constraint 2 (Demand Upper Bound): For each product i, the total produced units cannot exceed its upper demand: batch_size_units * x[i] ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality and Non-negativity): For each product i, x[i] ∈ {0, 1, 2, ...}.
[Abstract Model Plan END]