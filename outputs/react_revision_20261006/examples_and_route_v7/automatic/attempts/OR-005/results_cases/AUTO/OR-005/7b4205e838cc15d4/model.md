[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each distribution center (supplier) to each customer group, such that all customer demands are fully met, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (distribution centers): as listed in `supply_capacity.csv` (e.g., supplier1, supplier2, ..., supplier8)
    - Customers (customer groups): as listed in `customer_demand.csv` (e.g., demand1, demand2, ..., demand8)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods shipped from supplier `s` to customer `c` per day. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from each supplier to each customer: from `transportation_costs.csv` (columns: demand1...demand8, rows: supply1...supply8).
    -   Supply capacity for each supplier: from `supply_capacity.csv` (column: 'supply_capacity').
    -   Demand for each customer: from `customer_demand.csv` (column: 'demand').
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (quantity shipped from supplier to customer):  
    Minimize ∑ₛ ∑𝑐 [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group, the total quantity received from all suppliers must equal its demand.  
        For each customer `c`: ∑ₛ x[s, c] = customer_demand[c]
    -   Constraint 2 (Supply Capacity): For each supplier, the total quantity shipped to all customers must not exceed its supply capacity.  
        For each supplier `s`: ∑𝑐 x[s, c] ≤ supply_capacity[s]
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative.  
        For all suppliers `s` and customers `c`: x[s, c] ≥ 0
[Abstract Model Plan END]