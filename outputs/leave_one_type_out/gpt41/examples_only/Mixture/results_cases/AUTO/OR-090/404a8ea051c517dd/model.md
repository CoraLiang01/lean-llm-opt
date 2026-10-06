[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of production/purchase batches for each of 100 products (P1–P100), where each batch is 10 units, to maximize total profit. The solution must respect three resource capacity constraints and ensure that no product’s total produced units exceed its upper demand bound (unmet demand is allowed).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer batch decisions, linear constraints/objective).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, P2, ..., P100}), as listed in factory_products_100.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER (must be integer, as only whole batches are allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (profit per unit for each product), 'batch_size_units' (fixed at 10 for all products).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit for each product).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum allowed units per product), and 'capacity' for each resource from resources_capacities.csv.
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches) × (batch size) × (profit per unit), i.e., maximize sum over i of [x[i] * batch_size_units * profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource R1 Limit): The total R1 consumed by all produced units cannot exceed R1’s capacity. That is, sum over i of [x[i] * batch_size_units * r1_per_unit[i]] ≤ R1 capacity.
    -   Constraint 2 (Resource R2 Limit): sum over i of [x[i] * batch_size_units * r2_per_unit[i]] ≤ R2 capacity.
    -   Constraint 3 (Resource R3 Limit): sum over i of [x[i] * batch_size_units * r3_per_unit[i]] ≤ R3 capacity.
    -   Constraint 4 (Demand Upper Bound per Product): For each product i, total produced units cannot exceed upper_demand_units[i]. That is, x[i] * batch_size_units ≤ upper_demand_units[i].
    -   Constraint 5 (Non-negativity and Integrality): For each product i, x[i] ≥ 0 and integer (since only whole batches can be produced).
[Abstract Model Plan END]