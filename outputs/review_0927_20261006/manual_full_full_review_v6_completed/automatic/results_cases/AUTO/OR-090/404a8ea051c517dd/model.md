[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. Production is limited by three resource capacities and by each product’s upper demand (in units). Unmet demand is allowed, but overproduction is not. All resource and demand limits must be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with batch-size integer variables and resource constraints.
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce (each batch = 10 units). Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource usage per unit, from factory_products_100.csv).
    -   Constraint RHS: 'upper_demand_units' (per product, from factory_products_100.csv); 'capacity' for R1, R2, R3 (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches x[i]) × (batch size) × (profit per unit), i.e., maximize sum_i (10 × x[i] × profit_per_unit[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumption across all products cannot exceed its capacity:
        - sum_i (10 × x[i] × r{r}_per_unit[i]) ≤ capacity[r]
    -   Constraint 2 (Demand Upper Bound): For each product i, total produced units cannot exceed its upper demand:
        - 10 × x[i] ≤ upper_demand_units[i]
    -   Constraint 3 (Batch Integrality and Non-negativity): For each product i, x[i] ∈ {0, 1, 2, ...}
[Abstract Model Plan END]