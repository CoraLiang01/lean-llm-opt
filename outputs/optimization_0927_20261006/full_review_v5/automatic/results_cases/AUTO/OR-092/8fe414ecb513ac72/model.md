[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 sources to 20 destinations, using integer numbers of trucks (each with up to 10 units, but allowing partial loads), to meet all fixed demands at minimum total transportation cost. The cost is per unit shipped, and each source has a limited supply.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are Sources (S = {S1, ..., S10}) and Destinations (D = {D1, ..., D20}).
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER, t[s,d] ≥ 0.
    -   `x[s,d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS, 0 ≤ x[s,d] ≤ 10 * t[s,d].
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source s to destination d: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand at each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route s→d) × x[s,d].
7.  **Formulate Constraints:**
    -   Supply constraint: For each source s, the total units shipped out (sum over d of x[s,d]) ≤ supply_units[s].
    -   Demand constraint: For each destination d, the total units received (sum over s of x[s,d]) = demand_units[d].
    -   Truck loading constraint: For each (s,d), x[s,d] ≤ 10 × t[s,d].
    -   Truck integrality: For each (s,d), t[s,d] is integer and ≥ 0.
    -   Non-negativity: For each (s,d), x[s,d] ≥ 0.
[Abstract Model Plan END]