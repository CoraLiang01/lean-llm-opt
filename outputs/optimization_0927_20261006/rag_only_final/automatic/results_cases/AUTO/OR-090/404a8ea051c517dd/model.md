[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine optimal integer batch quantities for 100 products, where each batch is 10 units, to maximize total profit. Each product consumes three limited resources per unit produced, and production per product cannot exceed its upper demand bound (in units). Unmet demand is allowed. The solution must respect all resource capacities and batch-size constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed batch sizes and resource constraints.
3.  **Define Index Sets:** The primary index is Products (i ∈ set of all products in factory_products_100.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce or purchase. Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (profit per unit for each product), 'batch_size_units' (units per batch, fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit for R1, R2, R3).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum demand per product, in units), and 'capacity' from resources_capacities.csv for R1, R2, R3.
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (batch_size_units * x[i] * profit_per_unit[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource (R1, R2, R3), the total resource used across all products cannot exceed its capacity. For example, sum over i of (batch_size_units * x[i] * r1_per_unit[i]) ≤ capacity of R1, and similarly for R2 and R3.
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced (batch_size_units * x[i]) must not exceed upper_demand_units[i].
    -   Constraint 3 (Batch Integer): For each product i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]