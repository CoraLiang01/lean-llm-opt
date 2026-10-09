[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal quantities of products to ship from each warehouse to each store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are Warehouses (from 'region' in supply_capacity.csv) and Stores (from 'customer' in customer_demand.csv).
4.  **Define Decision Variables:**
    -   `x[w,s]` = Quantity of product shipped from warehouse `w` to store `s`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from warehouse to store, from 'transportation_costs.csv' (fields: warehouse as 'Unnamed: 0', stores as columns 'D1', 'D2', etc.).
    -   Constraint coefficients: Each unit shipped from warehouse `w` to store `s` counts toward both warehouse's supply and store's demand.
    -   Constraint RHS:
        -   Store demand: 'demand' column in 'customer_demand.csv' (key: 'customer').
        -   Warehouse supply capacity: 'supply_capacity' column in 'supply_capacity.csv' (key: 'region').
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and stores of (transportation cost per unit from warehouse to store) × (quantity shipped from warehouse to store).
7.  **Formulate Constraints:**
    -   Constraint 1 (Store Demand Satisfaction): For each store, the sum of shipments received from all warehouses must be at least equal to its demand (sum over warehouses of x[w,s] = demand[s] for each store s).
    -   Constraint 2 (Warehouse Supply Capacity): For each warehouse, the total quantity shipped out to all stores must not exceed its supply capacity (sum over stores of x[w,s] ≤ supply_capacity[w] for each warehouse w).
    -   Constraint 3 (Non-negativity): All shipment quantities x[w,s] ≥ 0.
[Abstract Model Plan END]