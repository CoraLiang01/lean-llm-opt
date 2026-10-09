[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products (P1–P100) in a single planning horizon, maximizing total profit, while ensuring that (a) total resource consumption for each of three resources (R1, R2, R3) does not exceed available capacities, and (b) the total produced units for each product do not exceed its upper demand bound. Each batch is 10 units, and only whole batches can be produced; unmet demand is allowed.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with batch-size integer variables and resource constraints.
3.  **Define Index Sets:** The primary indices are Products (i ∈ {P1, P2, ..., P100}) and Resources (r ∈ {R1, R2, R3}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches * batch size * profit per unit):  
    Maximize sum over i of [x[i] * batch_size_units * profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumption across all products cannot exceed the available capacity:  
        sum over i of [x[i] * batch_size_units * r{r}_per_unit[i]] ≤ capacity[r].
    -   Constraint 2 (Demand Upper Bound): For each product i, the total produced units cannot exceed its upper demand:  
        x[i] * batch_size_units ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality): For each product i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]