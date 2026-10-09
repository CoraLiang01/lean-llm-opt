[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum load of 10 units each, such that all demands are met, no source exceeds its supply, the number of trucks dispatched per route is an integer, and the total transportation cost (cost per unit of cargo) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) transportation problem with integer (truck count) variables and continuous (cargo per route) variables, due to the integer truck dispatch and partial loading.
3.  **Define Index Sets:** The primary indices are:
    -   Sources: S = {S1, S2, ..., S10} (from expanded_sources.csv)
    -   Destinations: D = {D1, D2, ..., D20} (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   `trucks[s, d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER (must be integer, ≥ 0).
    -   `cargo[s, d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS (≥ 0, can be fractional up to truck capacity × trucks[s, d]).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e.,  
    sum over all sources s and destinations d of (cost[s, d] * cargo[s, d]),  
    where cost[s, d] is the per-unit cost from expanded_cost_matrix.csv.
7.  **Formulate Constraints:**
    -   **Supply Constraint:** For each source s,  
        sum over all destinations d of cargo[s, d] ≤ supply_units[s]  
        (Do not ship more than available supply at each source.)
    -   **Demand Satisfaction:** For each destination d,  
        sum over all sources s of cargo[s, d] = demand_units[d]  
        (Exactly meet the demand at each destination.)
    -   **Truck Loading Constraint:** For each source s and destination d,  
        cargo[s, d] ≤ 10 × trucks[s, d]  
        (Cargo shipped on a route cannot exceed the total capacity of the dispatched trucks.)
    -   **Truck Integer Constraint:** For each source s and destination d,  
        trucks[s, d] ∈ {0, 1, 2, ...}  
        (Number of trucks dispatched must be integer and non-negative.)
    -   **Cargo Non-negativity:** For each source s and destination d,  
        cargo[s, d] ≥ 0  
        (Cannot ship negative cargo.)
[Abstract Model Plan END]