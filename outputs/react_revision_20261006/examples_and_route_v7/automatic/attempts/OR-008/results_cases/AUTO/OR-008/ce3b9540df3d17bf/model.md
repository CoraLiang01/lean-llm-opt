[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities of fresh produce from each warehouse (supplier) to each retail store (customer), such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses/Suppliers: {Supplier1, Supplier2, Supplier3, Supplier4, Supplier5} (from supply_capacity.csv and transportation_costs.csv)
    - Customers/Stores: {Customer1, Customer2, Customer3, Customer4, Customer5, Customer6} (from customer_demand.csv and transportation_costs.csv)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Amount of fresh produce shipped from supplier (warehouse) `s` to customer (store) `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from each supplier to each customer, from transportation_costs.csv (columns Customer1–Customer6, rows Supplier1–Supplier5).
    -   Constraint coefficients:
        -   Supply capacity per supplier: 'supply_capacity' column in supply_capacity.csv.
        -   Demand per customer: 'demand' column in customer_demand.csv.
    -   Constraint RHS (limits):
        -   For supply: Each supplier’s total shipments cannot exceed its 'supply_capacity'.
        -   For demand: Each customer must receive exactly its 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (amount shipped from supplier to customer):  
    Minimize  ∑ₛ ∑𝚌  [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Supply Capacity): For each supplier s, the total amount shipped out cannot exceed its supply capacity:  
        ∑𝚌 x[s, c] ≤ supply_capacity[s]  for all s in Suppliers.
    -   Constraint 2 (Demand Satisfaction): For each customer c, the total amount received from all suppliers must exactly meet its demand:  
        ∑ₛ x[s, c] = demand[c]  for all c in Customers.
    -   Constraint 3 (Non-negativity):  
        x[s, c] ≥ 0  for all suppliers s and customers c.
[Abstract Model Plan END]