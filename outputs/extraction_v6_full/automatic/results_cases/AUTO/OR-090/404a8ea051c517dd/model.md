[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce for each of 100 products, where each batch is 10 units, in order to maximize total profit. The plan must respect per-product upper demand limits (in units, not always a multiple of 10) and total available amounts of three resources (R1, R2, R3), given that each unit of each product consumes a specified amount of each resource. Unmet demand is allowed (i.e., not all demand must be satisfied), but overproduction is not.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary index is Products (i = 1 to 100, corresponding to P1–P100).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product i to produce (each batch = 10 units). Type: GRB.INTEGER, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'profit_per_unit' (profit per unit for each product), 'batch_size_units' (fixed at 10).
    -   Constraint coefficients: 'r1_per_unit', 'r2_per_unit', 'r3_per_unit' (resource consumption per unit for each product).
    -   Constraint RHS (limits): 'upper_demand_units' (maximum allowed units per product), and from resources_capacities.csv: 'capacity' for each resource (R1, R2, R3).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches produced) × (batch size, 10) × (profit per unit for that product):  
    Maximize sum over i of [10 × x[i] × profit_per_unit[i]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource (R1, R2, R3), the total resource consumed by all products cannot exceed the available capacity. For resource k (k ∈ {1,2,3}):  
        sum over i of [x[i] × 10 × r{k}_per_unit[i]] ≤ capacity[k].
    -   Constraint 2 (Demand Upper Bound): For each product i, the total units produced cannot exceed the upper demand for that product. Since production is in batches of 10 units:  
        x[i] × 10 ≤ upper_demand_units[i].
    -   Constraint 3 (Non-negativity and Integrality):  
        x[i] ≥ 0 and integer, for all products i.
[Abstract Model Plan END]