[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal allocation of goods from multiple distribution centers to multiple customer groups, such that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution Centers (Sources): S = {S1, S2, ..., S12} (from supply_capacity.csv and transportation_costs.csv, 'Unnamed: 0' column)
    - Customer Groups (Destinations): C = {C1, C2, ..., C12} (from customer_demand.csv and transportation_costs.csv, columns 'C1'...'C12')
4.  **Define Decision Variables:**
    -   `x[s, c]` = Amount of goods shipped from distribution center s ∈ S to customer group c ∈ C. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit (`cost[s, c]`): from transportation_costs.csv, entry at row s and column c.
    -   Supply capacity at each distribution center (`supply_capacity[s]`): from supply_capacity.csv, 'supply_capacity' column, keyed by 'Unnamed: 0' (distribution center ID).
    -   Demand for each customer group (`demand[c]`): from customer_demand.csv, 'demand' column, keyed by 'customer'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize sum over all s in S and c in C of (cost[s, c] * x[s, c]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group c ∈ C, the total goods received from all distribution centers must equal its demand:  
        sum over s in S of x[s, c] = demand[c]  ∀ c ∈ C.
    -   Constraint 2 (Supply Capacity): For each distribution center s ∈ S, the total goods shipped to all customer groups cannot exceed its supply capacity:  
        sum over c in C of x[s, c] ≤ supply_capacity[s]  ∀ s ∈ S.
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative:  
        x[s, c] ≥ 0  ∀ s ∈ S, c ∈ C.
[Abstract Model Plan END]