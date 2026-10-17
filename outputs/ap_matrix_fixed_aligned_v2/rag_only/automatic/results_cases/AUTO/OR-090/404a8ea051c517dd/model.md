[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of production/purchase batches for each of 100 products, where each batch is 10 units, to maximize total profit. Each product consumes three limited resources per unit produced, and cannot be produced beyond its upper demand (in units). Unmet demand is allowed. The solution must respect resource capacities and batch-size/demand interactions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed batch sizes and resource constraints.
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum allowed units per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
    -   Batch size: 'batch_size_units' (fixed at 10, from factory_products_100.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (batch_size_units * x[i] * profit_per_unit[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumption across all products cannot exceed its capacity:  
        sum over i of (batch_size_units * x[i] * r{r}_per_unit[i]) ≤ capacity[r].
    -   Constraint 2 (Demand Upper Bound): For each product i, the total produced units cannot exceed its upper demand:  
        batch_size_units * x[i] ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality): For each product i, x[i] is an integer ≥ 0.
[Abstract Model Plan END]