[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal allocation of goods from multiple suppliers to multiple customer groups, such that all customer demands are satisfied, no supplier exceeds its daily supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (S): as listed in `supply_capacity.csv` (e.g., S1, S2, ..., S10)
    - Customers (C): as listed in `customer_demand.csv` (e.g., C1, C2, ..., C10)
4.  **Define Decision Variables:**
    -   `x[s, c]` = quantity of goods transported from supplier `s` to customer `c`. Type: GRB.CONTINUOUS (non-negative real numbers; can be fractional if not otherwise restricted).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from supplier to customer: from `transportation_costs.csv`, columns C1–C10 for each supplier row.
    -   Supplier daily supply capacity: from `supply_capacity.csv`, column `supply_capacity` for each supplier.
    -   Customer daily demand: from `customer_demand.csv`, column `demand` for each customer.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (quantity shipped from supplier to customer):  
        Minimize ∑ₛ∈S ∑_c∈C [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Supply Capacity): For each supplier `s`, the total quantity shipped from supplier `s` to all customers cannot exceed its supply capacity:  
            ∑_c∈C x[s, c] ≤ supply_capacity[s]   for all s ∈ S
    -   Constraint 2 (Demand Satisfaction): For each customer `c`, the total quantity received from all suppliers must exactly meet its demand:  
            ∑ₛ∈S x[s, c] = demand[c]   for all c ∈ C
    -   Constraint 3 (Non-negativity):  
            x[s, c] ≥ 0   for all s ∈ S, c ∈ C
[Abstract Model Plan END]