[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to purchase/produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. Each unit produced consumes three limited resources, and production for each product cannot exceed its upper demand bound (unmet demand is allowed). All resource and demand limits must be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed batch sizes (integer multiples of 10 units per product).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER (non-negative).
        - The actual number of units produced for product i is `batch_size_units * x[i]` (with batch_size_units = 10 for all products).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv).
    -   Constraint coefficients:
        - Resource consumption per unit: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (from factory_products_100.csv).
        - Batch size: 'batch_size_units' (from factory_products_100.csv; always 10).
    -   Constraint RHS (limits):
        - Resource capacities: 'capacity' for each resource (from resources_capacities.csv).
        - Upper demand per product: 'upper_demand_units' (from factory_products_100.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of units produced) × (profit per unit), i.e., maximize sum over i of [batch_size_units × x[i] × profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total consumption across all products cannot exceed the available capacity:
        - sum over i of [batch_size_units × x[i] × r{r}_per_unit[i]] ≤ capacity[r]
        - (where r{r}_per_unit[i] is the per-unit consumption for resource r by product i)
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed its upper demand:
        - batch_size_units × x[i] ≤ upper_demand_units[i]
        - (since upper_demand_units[i] may not be a multiple of batch_size_units, this may force x[i] to be the largest integer such that batch_size_units × x[i] ≤ upper_demand_units[i])
    -   Constraint 3 (Non-negativity and Integrality): For all products i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]