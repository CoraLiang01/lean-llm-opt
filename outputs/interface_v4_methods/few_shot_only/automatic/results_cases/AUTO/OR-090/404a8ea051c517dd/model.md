[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. Production is limited by three resource capacities and by each product’s upper demand (in units). Unmet demand is allowed, but overproduction is not. All data is provided in two CSV files: one for product parameters and one for resource capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}). Resources (r ∈ {R1, R2, R3}) are used in constraints.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce (each batch = 10 units). Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource usage per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches produced) × (batch size) × (profit per unit), i.e., maximize sum over i of [10 × x[i] × profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumed by all products cannot exceed the available capacity. For example, for R1: sum over i of [10 × x[i] × r1_per_unit[i]] ≤ capacity_R1 (from resources_capacities.csv). Similarly for R2 and R3.
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed its upper demand, i.e., 10 × x[i] ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality): For each product i, x[i] ∈ {0, 1, 2, ...} (integer, non-negative).
[Abstract Model Plan END]