[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities of beverages from each production plant to each retail outlet, such that all customer demands are satisfied, no plant exceeds its production capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (Sources): S1, S2, S3, S4 (from `supply_capacity.csv`)
    - Customers (Destinations): C1, C2, C3, C4 (from `customer_demand.csv`)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of beverages shipped from plant `s` to customer `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from plant to customer: from `transportation_costs.csv`, columns `C1`, `C2`, `C3`, `C4` for each plant row.
    -   Plant supply capacity: from `supply_capacity.csv`, column `supply_capacity` for each plant.
    -   Customer demand: from `customer_demand.csv`, column `demand` for each customer.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all plants and customers of (transportation cost per unit from plant to customer) × (quantity shipped from plant to customer):  
    Minimize ∑ₛ ∑𝚌 [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer `c`, the total quantity received from all plants must equal the customer's demand:  
        ∑ₛ x[s, c] = customer_demand[c]  for all customers c.
    -   Constraint 2 (Supply Capacity): For each plant `s`, the total quantity shipped from that plant to all customers must not exceed the plant's supply capacity:  
        ∑𝚌 x[s, c] ≤ supply_capacity[s]  for all plants s.
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative:  
        x[s, c] ≥ 0  for all plants s and customers c.
[Abstract Model Plan END]