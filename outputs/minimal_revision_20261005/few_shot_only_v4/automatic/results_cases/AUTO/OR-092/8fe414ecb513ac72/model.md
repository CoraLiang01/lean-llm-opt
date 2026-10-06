[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal way to transport goods from 10 specified source locations (S1–S10) to 20 specified demand locations (D1–D20) using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are exactly met, no source exceeds its supply, and the total transportation cost (cost per unit, summed over all units shipped) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are:
    - Sources: S = {S1, S2, ..., S10} (from expanded_sources.csv, all rows)
    - Destinations: D = {D1, D2, ..., D20} (from expanded_destinations.csv, all rows)
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER (must be integer, can be zero).
    -   `f[s,d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS (0 ≤ f[s,d] ≤ 10 * t[s,d]).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from s to d: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand at each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all (s,d) of (cost per unit from s to d) × f[s,d].
7.  **Formulate Constraints:**
    -   Supply constraint (for each source s): The total amount shipped out from s cannot exceed its supply: sum over d of f[s,d] ≤ supply_units[s].
    -   Demand constraint (for each destination d): The total amount received at d must exactly meet its demand: sum over s of f[s,d] = demand_units[d].
    -   Truck loading constraint (for each (s,d)): The amount shipped on route (s,d) cannot exceed the total capacity of the dispatched trucks: f[s,d] ≤ 10 × t[s,d].
    -   Non-negativity and integrality: t[s,d] ≥ 0 and integer; f[s,d] ≥ 0 and continuous.
[Abstract Model Plan END]