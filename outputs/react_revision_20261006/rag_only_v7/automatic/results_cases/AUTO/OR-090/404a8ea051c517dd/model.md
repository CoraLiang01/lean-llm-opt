[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to purchase/produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. Each unit produced consumes three limited resources (R1, R2, R3), and each product has an upper demand bound (in units) that cannot be exceeded. Unmet demand is allowed. The solution must respect all resource capacities and demand limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed batch sizes (integer multiples of 10 units).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum allowed units per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches produced) × (batch size) × (profit per unit), i.e., maximize sum_i [x[i] * batch_size_units * profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumed by all products cannot exceed the available capacity. For example, sum_i [x[i] * batch_size_units * r1_per_unit[i]] ≤ capacity_R1, and similarly for R2 and R3.
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed the upper demand bound, i.e., x[i] * batch_size_units ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality): For each product i, x[i] is an integer ≥ 0 (cannot produce/purchase a fractional batch).
[Abstract Model Plan END]