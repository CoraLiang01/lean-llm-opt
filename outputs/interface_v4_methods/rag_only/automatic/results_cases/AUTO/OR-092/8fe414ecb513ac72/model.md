[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum load of 10 units each, such that all demands are met, no source exceeds its supply, the number of trucks dispatched per route is an integer, and the total transportation cost (cost per unit of cargo) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) transportation problem with integer (truck count) variables and continuous (cargo per route) variables, subject to supply, demand, and truck capacity constraints.
3.  **Define Index Sets:** The primary indices are:
    -   Sources: S = {S1, S2, ..., S10} (from expanded_sources.csv)
    -   Destinations: D = {D1, D2, ..., D20} (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   `trucks[s, d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER (must be whole trucks, ≥ 0).
    -   `cargo[s, d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS (0 ≤ cargo ≤ 10 × trucks[s, d]).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source to destination: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit from expanded_cost_matrix.csv) × (cargo[s, d]).
7.  **Formulate Constraints:**
    -   **Supply Constraint:** For each source s, the total cargo shipped out cannot exceed its supply:  
        sum over d of cargo[s, d] ≤ supply_units[s].
    -   **Demand Constraint:** For each destination d, the total cargo received must exactly meet its demand:  
        sum over s of cargo[s, d] = demand_units[d].
    -   **Truck Capacity Constraint:** For each route (s, d), the cargo shipped cannot exceed the total capacity of the dispatched trucks:  
        cargo[s, d] ≤ 10 × trucks[s, d].
    -   **Truck Integer Constraint:** For each route (s, d), trucks[s, d] must be an integer ≥ 0.
    -   **Cargo Nonnegativity:** For each route (s, d), cargo[s, d] ≥ 0.
[Abstract Model Plan END]