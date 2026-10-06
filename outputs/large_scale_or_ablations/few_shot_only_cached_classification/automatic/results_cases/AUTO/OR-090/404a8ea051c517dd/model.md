[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. The plan must respect per-product upper demand limits (in units) and total available capacities of three resources, given that each unit of each product consumes a specified amount of each resource. Unmet demand is allowed (i.e., not all demand must be met), but overproduction is not.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary index is Products (i = 1 to 100, corresponding to P1–P100). Resources (r = R1, R2, R3) are used in constraints.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce (each batch = 10 units). Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch_size_units (fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, i.e., maximize sum over all products of (number of batches produced) × (batch size) × (profit per unit):  
    Maximize: sum_{i} [ x[i] * batch_size_units * profit_per_unit[i] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumed by all products cannot exceed the available capacity:  
        sum_{i} [ x[i] * batch_size_units * r{r}_per_unit[i] ] ≤ capacity[r]
        (where r{r}_per_unit[i] is the per-unit consumption for resource r by product i)
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed the upper demand (but can be less):  
        x[i] * batch_size_units ≤ upper_demand_units[i]
        (since batch_size_units = 10, and upper_demand_units[i] may not be a multiple of 10, this may force x[i] to be the largest integer such that x[i] * 10 ≤ upper_demand_units[i])
    -   Constraint 3 (Non-negativity and Integrality):  
        x[i] ∈ {0, 1, 2, ...} for all i (integer, non-negative)
[Abstract Model Plan END]