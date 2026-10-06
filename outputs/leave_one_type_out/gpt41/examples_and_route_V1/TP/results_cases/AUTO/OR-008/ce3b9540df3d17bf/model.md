[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities of fresh produce from each warehouse (supplier) to each retail store (customer), such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses/Suppliers (from 'supply_capacity.csv' and 'transportation_costs.csv' rows): Supplier1, Supplier2, Supplier3, Supplier4, Supplier5.
    - Customers/Stores (from 'customer_demand.csv' and 'transportation_costs.csv' columns): Customer1, Customer2, Customer3, Customer4, Customer5, Customer6.
4.  **Define Decision Variables:**
    -   `x[s, c]` = Amount of fresh produce shipped from supplier (warehouse) `s` to customer (store) `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from each supplier to each customer, from 'transportation_costs.csv' (columns Customer1–Customer6, rows Supplier1–Supplier5).
    -   Constraint coefficients:
        -   Supply capacity per supplier: 'supply_capacity' column in 'supply_capacity.csv'.
        -   Demand per customer: 'demand' column in 'customer_demand.csv'.
    -   Constraint RHS:
        -   For supply: Each supplier's maximum supply (from 'supply_capacity').
        -   For demand: Each customer's required demand (from 'demand').
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (amount shipped from supplier to customer):  
    Minimize ∑<sub>s∈Suppliers</sub> ∑<sub>c∈Customers</sub> [transportation_costs[s, c] × x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer, the total amount received from all suppliers must equal that customer's demand.  
        For all c ∈ Customers: ∑<sub>s∈Suppliers</sub> x[s, c] = demand[c]
    -   Constraint 2 (Supply Capacity): For each supplier, the total amount shipped to all customers must not exceed that supplier's supply capacity.  
        For all s ∈ Suppliers: ∑<sub>c∈Customers</sub> x[s, c] ≤ supply_capacity[s]
    -   Constraint 3 (Non-negativity): All shipment variables must be non-negative.  
        For all s, c: x[s, c] ≥ 0
[Abstract Model Plan END]