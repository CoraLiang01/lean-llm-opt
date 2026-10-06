[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal way to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum capacity of 10 units per trip, such that all demands are met, no source exceeds its supply, only whole trucks are dispatched per route (integer number of trucks), and the total transportation cost (cost per unit of cargo, varying by route) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for partial truck loading.
3.  **Define Index Sets:** The primary indices are:
    -   Sources: S = {S1, S2, ..., S10} (from expanded_sources.csv)
    -   Destinations: D = {D1, D2, ..., D20} (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER (must be whole trucks, can be zero).
    -   `q[s,d]` = Quantity of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS (can be any value between 0 and 10 × t[s,d]).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source to destination: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units'.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (problem statement).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit for route s→d) × (quantity shipped q[s,d]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Truck Loading): For each route (s,d), the quantity shipped cannot exceed the total capacity of the dispatched trucks:  
        q[s,d] ≤ 10 × t[s,d]
    -   Constraint 2 (Supply Limit): For each source s, the total quantity shipped from s cannot exceed its available supply:  
        sum over d of q[s,d] ≤ supply_units[s]
    -   Constraint 3 (Demand Satisfaction): For each destination d, the total quantity received must exactly meet its demand:  
        sum over s of q[s,d] = demand_units[d]
    -   Constraint 4 (Truck Integer): For each route (s,d), t[s,d] must be an integer ≥ 0.
    -   Constraint 5 (Non-negativity): For each route (s,d), q[s,d] ≥ 0.
[Abstract Model Plan END]