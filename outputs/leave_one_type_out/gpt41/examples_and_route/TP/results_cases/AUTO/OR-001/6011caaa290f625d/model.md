[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily transportation plan from multiple Amazon distribution centers to various customer groups, such that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution Centers (Sources): S = {S1, S2, ..., S18} (from supply_capacity.csv and transportation_costs.csv, 'Unnamed: 0' column)
    - Customer Groups (Destinations): C = {C1, C2, ..., C18} (from customer_demand.csv and transportation_costs.csv, columns 'C1'...'C18')
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported daily from distribution center s ∈ S to customer group c ∈ C. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit (`cost[s, c]`): from transportation_costs.csv, entry at row s and column c.
    -   Supply capacity at each distribution center (`supply_capacity[s]`): from supply_capacity.csv, 'supply_capacity' column, keyed by 'Unnamed: 0' (S1...S18).
    -   Demand for each customer group (`demand[c]`): from customer_demand.csv, 'demand' column, keyed by 'customer' (C1...C18).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize sum over all s in S and c in C of (cost[s, c] * x[s, c]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group c ∈ C, the total goods received from all distribution centers must equal its demand:  
        sum over s in S of x[s, c] = demand[c].
    -   Constraint 2 (Supply Capacity): For each distribution center s ∈ S, the total goods shipped to all customer groups cannot exceed its supply capacity:  
        sum over c in C of x[s, c] ≤ supply_capacity[s].
    -   Constraint 3 (Non-negativity): For all s ∈ S and c ∈ C, x[s, c] ≥ 0.
[Abstract Model Plan END]