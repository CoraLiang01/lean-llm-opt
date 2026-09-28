[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily transportation plan for delivering goods from multiple suppliers to multiple customer groups, such that all customer demands are satisfied, no supplier exceeds its daily capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (S): as listed in `supply_capacity.csv` and `transportation_costs.csv` (e.g., S1, S2, ..., S10)
    - Customers (C): as listed in `customer_demand.csv` and `transportation_costs.csv` (e.g., C1, C2, ..., C10)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Amount of goods transported from supplier `s` to customer `c` per day. Type: GRB.CONTINUOUS (non-negative real numbers; fractional shipments allowed).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from supplier to customer: from `transportation_costs.csv`, columns `C1`...`C10` for each supplier row.
    -   Supplier daily capacity: from `supply_capacity.csv`, column `supply_capacity` for each supplier.
    -   Customer daily demand: from `customer_demand.csv`, column `demand` for each customer.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (amount shipped from supplier to customer):  
        Minimize ∑ₛ∈S ∑_c∈C [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Supply Capacity): For each supplier, the total amount shipped out does not exceed its daily capacity:  
            For all s ∈ S:  ∑_c∈C x[s, c] ≤ supply_capacity[s]
    -   Constraint 2 (Demand Satisfaction): For each customer, the total amount received from all suppliers exactly meets its daily demand:  
            For all c ∈ C:  ∑ₛ∈S x[s, c] = demand[c]
    -   Constraint 3 (Non-negativity):  
            For all s ∈ S, c ∈ C:  x[s, c] ≥ 0
[Abstract Model Plan END]