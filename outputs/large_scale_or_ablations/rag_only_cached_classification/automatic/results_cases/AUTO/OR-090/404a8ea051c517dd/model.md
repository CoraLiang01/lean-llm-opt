[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of production/purchase batches for each of 100 products, where each batch is 10 units, to maximize total profit. Each product consumes three limited resources per unit produced, and cannot be produced beyond its upper demand (in units). Unmet demand is allowed. The solution must respect resource capacities and batch-size/demand interactions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (specifically, a multi-resource, multi-product, batch-size-constrained knapsack-type model).
3.  **Define Index Sets:** The primary index is Products (i ∈ {P1, ..., P100}). Resources (r ∈ {R1, R2, R3}) are used as parameters in constraints.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), batch size ('batch_size_units' = 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum allowed units per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (batch_size_units * x[i] * profit_per_unit[i]). That is, for each product, profit per unit × units produced (which is batch_size_units × x[i]), summed over all products.
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total consumption across all products cannot exceed the available capacity. For each resource, sum over all products of (batch_size_units × x[i] × r{r}_per_unit[i]) ≤ capacity[r].
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced (batch_size_units × x[i]) cannot exceed upper_demand_units[i]. That is, batch_size_units × x[i] ≤ upper_demand_units[i].
    -   Constraint 3 (Batch Integrality): For each product i, x[i] must be a non-negative integer (x[i] ≥ 0, integer).
[Abstract Model Plan END]