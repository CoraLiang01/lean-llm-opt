[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized. The relevant data includes daily customer demands, store supply capacities, and per-unit transportation costs between each store and customer group.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (sources): Identified by 'Unnamed: 0' in supply_capacity.csv and transportation_costs.csv (e.g., S1, S2, ..., S11).
    - Customers (destinations): Identified by 'customer' in customer_demand.csv and as columns C1–C12 in transportation_costs.csv.
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store s to customer group c. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Per-unit transportation costs from 'transportation_costs.csv' (columns C1–C12 for each store).
    -   Constraint coefficients:
        -   For supply constraints: Each store's total outgoing shipments, using 'supply_capacity' from supply_capacity.csv.
        -   For demand constraints: Each customer's total incoming shipments, using 'demand' from customer_demand.csv.
    -   Constraint RHS (limits):
        -   Supply limits: 'supply_capacity' for each store.
        -   Demand requirements: 'demand' for each customer group.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all stores and customer groups of (transportation cost per unit from store s to customer c) × (quantity shipped from s to c):  
    Minimize ∑ₛ ∑_c [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group c, the sum of shipments received from all stores must equal that group's demand:  
        ∑ₛ x[s, c] = customer_demand[c]
    -   Constraint 2 (Supply Capacity): For each store s, the sum of shipments sent to all customer groups must not exceed that store's supply capacity:  
        ∑_c x[s, c] ≤ supply_capacity[s]
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative:  
        x[s, c] ≥ 0 for all s, c
[Abstract Model Plan END]