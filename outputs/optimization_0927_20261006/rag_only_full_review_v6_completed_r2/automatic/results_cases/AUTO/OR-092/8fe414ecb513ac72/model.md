[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how to transport goods from 10 source locations to 20 demand locations using trucks with a maximum capacity of 10 units each, such that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit of cargo) is minimized. The number of trucks dispatched per route must be integer, but truck loads can be partial.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) transportation problem with integer truck dispatches and continuous cargo allocation per truck.
3.  **Define Index Sets:** The primary indices are:
    - Sources (S): All rows from expanded_sources.csv, identified by 'source_id'.
    - Destinations (D): All rows from expanded_destinations.csv, identified by 'destination_id'.
4.  **Define Decision Variables:**
    -   `trucks[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER, ≥ 0.
    -   `cargo[s,d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS, ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from expanded_cost_matrix.csv, columns ['D1', ..., 'D20'] for each 'source_id'.
    -   Supply limits: from expanded_sources.csv, column 'supply_units' for each 'source_id'.
    -   Demand requirements: from expanded_destinations.csv, column 'demand_units' for each 'destination_id'.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit from expanded_cost_matrix.csv) × cargo[s,d].
7.  **Formulate Constraints:**
    -   Supply constraint: For each source s, sum over all destinations d of cargo[s,d] ≤ supply_units[s].
    -   Demand constraint: For each destination d, sum over all sources s of cargo[s,d] = demand_units[d].
    -   Truck loading constraint: For each (s,d), cargo[s,d] ≤ 10 × trucks[s,d].
    -   Truck integrality: For each (s,d), trucks[s,d] ∈ {0, 1, 2, ...}.
    -   Cargo non-negativity: For each (s,d), cargo[s,d] ≥ 0.
[Abstract Model Plan END]