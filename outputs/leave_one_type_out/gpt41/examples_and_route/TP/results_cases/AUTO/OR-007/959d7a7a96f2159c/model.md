[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal quantities of products to ship from each warehouse to each GreenMart store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from 'supply_capacity.csv', column 'region', e.g., S1, S2, S3, S4, S5)
    - Stores (from 'customer_demand.csv', column 'customer', e.g., D1, D2, D3, D4, D5)
4.  **Define Decision Variables:**
    -   `x[w,s]` = Quantity of product shipped from warehouse `w` to store `s`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from warehouse to store: from 'transportation_costs.csv', columns D1–D5, rows S1–S5.
    -   Warehouse supply capacity: from 'supply_capacity.csv', column 'supply_capacity' for each 'region'.
    -   Store demand: from 'customer_demand.csv', column 'demand' for each 'customer'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and stores of (transportation cost per unit from warehouse to store) × (quantity shipped from warehouse to store):  
        Minimize ∑₍w,s₎ [transportation_cost[w,s] * x[w,s]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Store Demand Satisfaction): For each store `s`, the total quantity received from all warehouses must be at least its demand:  
            ∑₍w₎ x[w,s] = demand[s]   for all stores s
    -   Constraint 2 (Warehouse Supply Capacity): For each warehouse `w`, the total quantity shipped out to all stores cannot exceed its supply capacity:  
            ∑₍s₎ x[w,s] ≤ supply_capacity[w]   for all warehouses w
    -   Constraint 3 (Non-negativity):  
            x[w,s] ≥ 0   for all warehouses w and stores s
[Abstract Model Plan END]