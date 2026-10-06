[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for each of 100 products, how many whole batches (of 10 units each) to purchase/produce in order to maximize total profit, while ensuring that (a) total consumption of each of three resources does not exceed available capacities, and (b) the total produced quantity for each product does not exceed its upper demand bound (unmet demand is allowed). All data is provided in two CSV files: one for product parameters and one for resource capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with integer (batch) variables and linear constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: All 100 products listed in factory_products_100.csv (indexed by i).
    - Resources: The three resources R1, R2, R3 from resources_capacities.csv (indexed by r).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase. Type: GRB.INTEGER, x[i] ≥ 0.
        - Each batch is exactly 10 units (batch_size_units = 10 for all products).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - 'profit_per_unit' (from factory_products_100.csv): profit earned per unit of product i.
        - 'batch_size_units' (from factory_products_100.csv): fixed at 10 for all products.
    -   Constraint coefficients:
        - 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (from factory_products_100.csv): resource consumption per unit for each product.
    -   Constraint RHS (limits):
        - 'upper_demand_units' (from factory_products_100.csv): maximum allowed units produced for each product.
        - 'capacity' (from resources_capacities.csv): total available amount for each resource.
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches produced) × (batch size) × (profit per unit), i.e., maximize sum over i of [x[i] * batch_size_units * profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r (R1, R2, R3), the total consumption across all products cannot exceed the resource's capacity:
        - sum over i of [x[i] * batch_size_units * r{r}_per_unit[i]] ≤ capacity[r]
        - (where r{r}_per_unit[i] is the per-unit consumption for resource r by product i)
    -   Constraint 2 (Demand Upper Bound): For each product i, the total produced units cannot exceed its upper demand:
        - x[i] * batch_size_units ≤ upper_demand_units[i]
    -   Constraint 3 (Batch Integrality): For each product i, x[i] is an integer ≥ 0 (cannot produce/purchase a fractional batch).
[Abstract Model Plan END]