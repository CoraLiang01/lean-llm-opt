[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities from each production plant to each retail outlet, such that all customer demands are satisfied, no plant exceeds its production capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (sources): as listed in `supply_capacity.csv` (e.g., S1, S2, S3, S4)
    - Customers (destinations): as listed in `customer_demand.csv` (e.g., C1, C2, C3, C4)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of beverages shipped from plant `s` to customer `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from plant to customer: from `transportation_costs.csv`, columns `C1`, `C2`, `C3`, `C4` indexed by plant (`Unnamed: 0`).
    -   Plant production capacities: from `supply_capacity.csv`, column `supply_capacity` indexed by plant (`Unnamed: 0`).
    -   Customer demands: from `customer_demand.csv`, column `demand` indexed by customer (`customer`).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all plants and customers of (transportation cost per unit from plant to customer) × (quantity shipped from plant to customer):  
    Minimize ∑ₛ ∑𝚌 [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer, the total quantity received from all plants must equal that customer's demand:  
        For all customers c: ∑ₛ x[s, c] = customer_demand[c]
    -   Constraint 2 (Supply Capacity): For each plant, the total quantity shipped to all customers must not exceed the plant's supply capacity:  
        For all plants s: ∑𝚌 x[s, c] ≤ supply_capacity[s]
    -   Constraint 3 (Non-negativity): For all plants s and customers c: x[s, c] ≥ 0
[Abstract Model Plan END]