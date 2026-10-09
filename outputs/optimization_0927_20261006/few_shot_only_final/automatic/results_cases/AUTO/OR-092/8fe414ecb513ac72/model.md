[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of truck trips (each up to 10 units, partial loading allowed) from each of 10 sources to each of 20 destinations, so that all destination demands are exactly met, no source exceeds its supply, and total transportation cost (cost per unit, summed over all units shipped) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and partial truck loading allowed.
3.  **Define Index Sets:** The primary indices are Sources (S = {S1, ..., S10}) and Destinations (D = {D1, ..., D20}).
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER, t[s,d] ≥ 0.
    -   `q[s,d]` = Quantity of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS, 0 ≤ q[s,d] ≤ 10 * t[s,d].
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from expanded_cost_matrix.csv, field [source_id][destination_id].
    -   Source supply limits: from expanded_sources.csv, field 'supply_units' for each source_id.
    -   Destination demand: from expanded_destinations.csv, field 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route s→d) × q[s,d].
7.  **Formulate Constraints:**
    -   Supply constraint: For each source s, the total quantity shipped out (sum over d of q[s,d]) ≤ supply_units[s].
    -   Demand constraint: For each destination d, the total quantity received (sum over s of q[s,d]) = demand_units[d].
    -   Truck loading constraint: For each (s,d), q[s,d] ≤ 10 × t[s,d].
    -   Truck integrality: For each (s,d), t[s,d] ∈ {0,1,2,...}.
    -   Nonnegativity: For each (s,d), q[s,d] ≥ 0.
[Abstract Model Plan END]