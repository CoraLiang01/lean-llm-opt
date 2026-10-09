[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal quantities of products to ship from each warehouse to each retail store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from `supply_capacity.csv`, keys: S1, S2, ..., S10)
    - Customers/Stores (from `customer_demand.csv`, keys: C1, C2, ..., C10)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of product shipped from warehouse `s` to customer/store `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from warehouse to store, from `transportation_costs.csv` (columns C1–C10, rows S1–S10).
    -   Constraint coefficients:
        -   Warehouse supply capacities: from `supply_capacity.csv` (column `supply_capacity` for each warehouse).
        -   Customer/store demands: from `customer_demand.csv` (column `demand` for each customer).
    -   Constraint RHS (limits): 
        -   For warehouses: their supply capacity.
        -   For customers: their demand.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and customers of (transportation cost per unit from warehouse to customer) × (quantity shipped from warehouse to customer):  
    Minimize ∑ₛ ∑_c [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer/store c, the total quantity received from all warehouses must be at least (or exactly) their demand:  
        ∑ₛ x[s, c] = customer_demand[c]
    -   Constraint 2 (Supply Capacity): For each warehouse s, the total quantity shipped out to all customers cannot exceed its supply capacity:  
        ∑_c x[s, c] ≤ supply_capacity[s]
    -   Constraint 3 (Non-negativity): For all s, c:  
        x[s, c] ≥ 0
[Abstract Model Plan END]