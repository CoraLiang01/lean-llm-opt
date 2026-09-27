[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. Production is limited by three resource capacities and by each product’s upper demand (in units). Unmet demand is allowed, but overproduction is not. All data is provided in two CSV files: one with product-level parameters and one with resource capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}). Resources (r ∈ {R1, R2, R3}) are used in constraints.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce (each batch = 10 units). Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource usage per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, i.e., maximize the sum over all products of (number of batches produced) × (batch size) × (profit per unit):  
    Maximize sum over i of [x[i] * batch_size_units * profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumed by all products cannot exceed the available capacity. For each r:  
        sum over i of [x[i] * batch_size_units * r{r}_per_unit[i]] ≤ capacity[r].
    -   Constraint 2 (Demand Upper Bound): For each product i, total units produced cannot exceed upper_demand_units[i]. Since production is in batches of 10 units:  
        x[i] * batch_size_units ≤ upper_demand_units[i].
    -   Constraint 3 (Non-negativity and Integrality): For all i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]