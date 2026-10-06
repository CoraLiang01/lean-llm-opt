[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily transportation plan from multiple Amazon distribution centers to various customer groups, such that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution Centers (Sources): S = {S1, S2, ..., S18} (from 'supply_capacity.csv' and 'transportation_costs.csv' rows)
    - Customer Groups (Destinations): C = {C1, C2, ..., C18} (from 'customer_demand.csv' and 'transportation_costs.csv' columns)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from distribution center s ∈ S to customer group c ∈ C. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from s to c: from 'transportation_costs.csv', entry at row s, column c.
    -   Supply capacity at each distribution center s: from 'supply_capacity.csv', column 'supply_capacity', keyed by 'Unnamed: 0' (distribution center ID).
    -   Demand for each customer group c: from 'customer_demand.csv', column 'demand', keyed by 'customer'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all s and c of (transportation cost from s to c) × x[s, c].
7.  **Formulate Constraints:**
    -   Constraint 1 (Supply Capacity at Each Distribution Center): For each s ∈ S, the total quantity shipped out from s to all customers cannot exceed its supply capacity:
        - sum over c of x[s, c] ≤ supply_capacity[s]
    -   Constraint 2 (Demand Satisfaction for Each Customer Group): For each c ∈ C, the total quantity received from all distribution centers must exactly meet the demand:
        - sum over s of x[s, c] = demand[c]
    -   Constraint 3 (Non-negativity): For all s ∈ S and c ∈ C, x[s, c] ≥ 0
[Abstract Model Plan END]