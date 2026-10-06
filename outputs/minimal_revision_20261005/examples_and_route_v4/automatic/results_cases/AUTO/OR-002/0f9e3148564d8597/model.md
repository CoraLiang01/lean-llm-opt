[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (sources): S1, S2, ..., S11 (from supply_capacity.csv and transportation_costs.csv)
    - Customers (destinations): C1, C2, ..., C12 (from customer_demand.csv and transportation_costs.csv)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store `s` to customer group `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from each store to each customer, from 'transportation_costs.csv' (columns C1–C12, rows S1–S11).
    -   Constraint coefficients:
        -   Supply capacity per store: 'supply_capacity' column in 'supply_capacity.csv' (rows S1–S11).
        -   Demand per customer: 'demand' column in 'customer_demand.csv' (rows C1–C12).
    -   Constraint RHS:
        -   For supply: Each store's supply capacity.
        -   For demand: Each customer's demand.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all stores and customers of (transportation cost per unit from store s to customer c) × (quantity shipped from s to c):  
    Minimize ∑ₛ∈Stores ∑_c∈Customers [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group c, the total quantity received from all stores must equal its demand:  
        ∑ₛ∈Stores x[s, c] = customer_demand[c]  for all c ∈ Customers
    -   Constraint 2 (Supply Capacity): For each store s, the total quantity shipped to all customers must not exceed its supply capacity:  
        ∑_c∈Customers x[s, c] ≤ supply_capacity[s]  for all s ∈ Stores
    -   Constraint 3 (Non-negativity):  
        x[s, c] ≥ 0  for all s ∈ Stores, c ∈ Customers
[Abstract Model Plan END]