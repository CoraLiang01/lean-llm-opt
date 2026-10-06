[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of production/purchase batches for each of 100 products (P1–P100), where each batch is 10 units, to maximize total profit. The solution must not exceed available capacities of three resources (R1, R2, R3), and for each product, total produced units cannot exceed its upper demand bound (unmet demand is allowed).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer batch decisions, linear constraints/objective).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}), as listed in factory_products_100.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER (must be integer, as only whole batches are allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (profit per unit for each product), 'batch_size_units' (units per batch, always 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit for each product).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum allowed units per product), and from resources_capacities.csv: 'capacity' for each resource (R1, R2, R3).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches) × (units per batch) × (profit per unit), i.e., maximize sum over i of [x[i] * batch_size_units * profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource (R1, R2, R3), the total resource consumed by all products cannot exceed its available capacity. For example, for R1: sum over i of [x[i] * batch_size_units * r1_per_unit[i]] ≤ R1_capacity (from resources_capacities.csv). Similarly for R2 and R3.
    -   Constraint 2 (Demand Upper Bound): For each product i, the total produced units cannot exceed its upper demand, i.e., x[i] * batch_size_units ≤ upper_demand_units[i].
    -   Constraint 3 (Non-negativity and Integrality): For each product i, x[i] ≥ 0 and integer (since only whole batches can be produced/purchased).
[Abstract Model Plan END]