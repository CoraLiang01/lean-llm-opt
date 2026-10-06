[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products, where each batch is 10 units, in order to maximize total profit. Production of each product consumes three limited resources, and each product has an upper demand bound (in units, not necessarily a multiple of 10). The solution must respect all resource capacities and not exceed demand for any product.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary indices are Products (P1–P100) and Resources (R1, R2, R3).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce/purchase (where i ∈ Products). Type: GRB.INTEGER (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (from factory_products_100.csv), multiplied by batch size (10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit, from factory_products_100.csv), multiplied by batch size (10).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum units per product, from factory_products_100.csv); 'capacity' for each resource (from resources_capacities.csv).
    -   'batch_size_units' is fixed at 10 for all products.
6.  **Formulate Objective:** Maximize total profit, i.e., maximize the sum over all products of (number of batches of product i) × (batch size) × (profit per unit for i):  
    Maximize: sum over i of [10 × x[i] × profit_per_unit[i]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource r ∈ {R1, R2, R3}, the total resource consumption across all products cannot exceed the available capacity:  
        For each r: sum over i of [10 × x[i] × r{r}_per_unit[i]] ≤ capacity[r]
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed the upper demand (note: only full batches can be produced, so the actual units produced is 10 × x[i]):  
        For each i: 10 × x[i] ≤ upper_demand_units[i]
    -   Constraint 3 (Non-negativity and Integrality): For each product i, x[i] ≥ 0 and integer.
[Abstract Model Plan END]