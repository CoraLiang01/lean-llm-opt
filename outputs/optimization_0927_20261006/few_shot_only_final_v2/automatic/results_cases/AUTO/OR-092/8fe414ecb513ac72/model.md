[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of truck trips (each up to 10 units, partial loading allowed) from each of 10 sources to each of 20 destinations, so that all destination demands are exactly met, no source exceeds its supply, and total transportation cost (per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and partial truck loading allowed.
3.  **Define Index Sets:** The primary indices are Sources (S = {S1, ..., S10}) and Destinations (D = {D1, ..., D20}).
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d (integer, ≥ 0). Type: GRB.INTEGER.
    -   `q[s,d]` = Quantity of cargo (units) shipped from source s to destination d (continuous, 0 ≤ q[s,d] ≤ 10 * t[s,d]). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit for each route: from expanded_cost_matrix.csv, columns 'source_id', 'D1'...'D20'.
    -   Source supply limits: from expanded_sources.csv, columns 'source_id', 'supply_units'.
    -   Destination demand requirements: from expanded_destinations.csv, columns 'destination_id', 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route s→d) × q[s,d].
7.  **Formulate Constraints:**
    -   Supply constraint: For each source s, the total quantity shipped out (sum over all d of q[s,d]) ≤ supply_units[s].
    -   Demand constraint: For each destination d, the total quantity received (sum over all s of q[s,d]) = demand_units[d].
    -   Truck loading constraint: For each (s,d), q[s,d] ≤ 10 × t[s,d].
    -   Truck integer constraint: For each (s,d), t[s,d] ∈ {0, 1, 2, ...}.
    -   Nonnegativity: For each (s,d), q[s,d] ≥ 0.
[Abstract Model Plan END]