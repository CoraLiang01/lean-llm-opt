[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal quantities of products to ship from each warehouse to each GreenMart store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are Warehouses (from 'region' in supply_capacity.csv) and Stores (from 'customer' in customer_demand.csv).
4.  **Define Decision Variables:**
    -   `x[s, d]` = Quantity of product shipped from warehouse (region) `s` to store (customer) `d`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from warehouse to store, from 'transportation_costs.csv' columns D1–D5 (indexed by warehouse and store).
    -   Constraint coefficients: Each unit shipped counts toward both warehouse supply and store demand.
    -   Constraint RHS (limits): 
        -   Store demand from 'demand' in customer_demand.csv.
        -   Warehouse supply capacity from 'supply_capacity' in supply_capacity.csv.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and stores of (transportation cost per unit from warehouse to store) × (quantity shipped from warehouse to store).
7.  **Formulate Constraints:**
    -   Constraint 1 (Store Demand Satisfaction): For each store, the sum of shipments received from all warehouses must be at least equal to its demand (sum over warehouses of x[s, d] = demand[d] for each store d).
    -   Constraint 2 (Warehouse Supply Capacity): For each warehouse, the total quantity shipped to all stores must not exceed its supply capacity (sum over stores of x[s, d] ≤ supply_capacity[s] for each warehouse s).
    -   Constraint 3 (Non-negativity): All shipment quantities x[s, d] ≥ 0.
[Abstract Model Plan END]