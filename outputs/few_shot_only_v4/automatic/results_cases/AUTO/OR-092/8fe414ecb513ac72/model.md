[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of truck trips from each of 10 sources to each of 20 destinations, where each truck can carry up to 10 units (partial loading allowed), to satisfy all fixed destination demands without exceeding source supplies, and to minimize total transportation cost (cost per unit shipped, summed over all routes).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) transportation problem with integer variables for truck trips and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are Sources (S = {S1, ..., S10}) and Destinations (D = {D1, ..., D20}).
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER (nonnegative).
    -   `x[s,d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS (0 ≤ x[s,d] ≤ 10 * t[s,d]).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from expanded_cost_matrix.csv, field at [source_id, destination_id].
    -   Source supply limits: from expanded_sources.csv, field 'supply_units' for each source_id.
    -   Destination demand: from expanded_destinations.csv, field 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route s→d) × x[s,d].
7.  **Formulate Constraints:**
    -   Supply constraint: For each source s, sum over all destinations d of x[s,d] ≤ supply_units[s].
    -   Demand constraint: For each destination d, sum over all sources s of x[s,d] = demand_units[d].
    -   Truck loading constraint: For each (s,d), x[s,d] ≤ 10 × t[s,d].
    -   Nonnegativity and integrality: For all (s,d), t[s,d] ≥ 0 and integer; x[s,d] ≥ 0 and continuous.
[Abstract Model Plan END]