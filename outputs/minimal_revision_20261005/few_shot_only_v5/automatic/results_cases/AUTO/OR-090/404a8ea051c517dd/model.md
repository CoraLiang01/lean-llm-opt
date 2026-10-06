[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products (P1–P100) in a single planning horizon, where each batch is 10 units, to maximize total profit. Production of each product consumes three limited resources (R1, R2, R3), and each product has an upper demand bound (in units) that cannot be exceeded. Unmet demand is allowed (i.e., not all demand must be met), but production must not exceed the upper bound. All resource and demand constraints must be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER (non-negative).
        - Each batch is exactly 10 units (batch_size_units = 10 for all products).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch_size_units (fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv), 'capacity' for R1, R2, R3 (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, which is the sum over all products of (number of batches produced) × (batch size in units) × (profit per unit for that product). That is, maximize sum over i of [10 × x[i] × profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total consumption across all products cannot exceed the available capacity:
        - sum over i of [x[i] × 10 × r{r}_per_unit[i]] ≤ capacity[r]
        - Specifically:
            - sum_i [x[i] × 10 × r1_per_unit[i]] ≤ capacity[R1]
            - sum_i [x[i] × 10 × r2_per_unit[i]] ≤ capacity[R2]
            - sum_i [x[i] × 10 × r3_per_unit[i]] ≤ capacity[R3]
    -   Constraint 2 (Demand Upper Bound): For each product i, total units produced cannot exceed its upper demand:
        - x[i] × 10 ≤ upper_demand_units[i]
        - Since upper_demand_units[i] may not be a multiple of 10, this may force x[i] ≤ floor(upper_demand_units[i] / 10)
    -   Constraint 3 (Non-negativity and Integrality): For all i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]