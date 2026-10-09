[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of production/purchase batches for each of 100 products, where each batch is 10 units, to maximize total profit. Each product consumes three limited resources per unit produced, and cannot be produced beyond its upper demand (in units). Unmet demand is allowed. The solution must respect resource capacities and batch-size/demand interactions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed batch sizes and resource constraints.
3.  **Define Index Sets:** The primary index is Products (i ∈ set of 100 products from 'factory_products_100.csv'). Resources (r ∈ {R1, R2, R3}) are used in constraints.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (profit per unit for each product), 'batch_size_units' (units per batch, fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit for each product).
    -   Constraint RHS: 'upper_demand_units' (maximum allowed units per product), 'capacity' from 'resources_capacities.csv' (total available for each resource).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (batch_size_units * x[i] * profit_per_unit[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumed by all products cannot exceed its capacity:  
        sum over i of (batch_size_units * x[i] * r{r}_per_unit[i]) ≤ capacity[r].
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed its upper demand:  
        batch_size_units * x[i] ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality): For each product i, x[i] ∈ {0, 1, 2, ...} (integer, non-negative).
[Abstract Model Plan END]