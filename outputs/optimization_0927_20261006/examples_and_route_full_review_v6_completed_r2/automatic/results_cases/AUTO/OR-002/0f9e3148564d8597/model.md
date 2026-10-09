[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores (suppliers) to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized. All required data is provided in three CSV files: customer demands, store supply capacities, and per-unit transportation costs between each store and customer group.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (S): Each Walmart store, as listed in `supply_capacity.csv` and `transportation_costs.csv` (e.g., S1, S2, ..., S11).
    - Customers (C): Each customer group, as listed in `customer_demand.csv` and `transportation_costs.csv` (e.g., C1, C2, ..., C12).
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store `s` to customer group `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from store to customer: from `transportation_costs.csv`, columns `C1`–`C12` for each row (store).
    -   Supply capacity for each store: from `supply_capacity.csv`, column `supply_capacity`.
    -   Demand for each customer group: from `customer_demand.csv`, column `demand`.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all stores and customer groups of (transportation cost per unit from store to customer) × (quantity transported from that store to that customer):  
    Minimize ∑ₛ∈S ∑𝚌∈C [transportation_costs[s, c] × x[s, c]].
7.  **Formulate Constraints:**
    -   Supply Capacity Constraint (for each store): The total quantity shipped from each store to all customer groups cannot exceed that store’s supply capacity:  
        ∑𝚌∈C x[s, c] ≤ supply_capacity[s], ∀ s ∈ S.
    -   Demand Satisfaction Constraint (for each customer group): The total quantity received by each customer group from all stores must exactly meet that group’s demand:  
        ∑ₛ∈S x[s, c] = demand[c], ∀ c ∈ C.
    -   Non-negativity: x[s, c] ≥ 0 for all s ∈ S, c ∈ C.
[Abstract Model Plan END]