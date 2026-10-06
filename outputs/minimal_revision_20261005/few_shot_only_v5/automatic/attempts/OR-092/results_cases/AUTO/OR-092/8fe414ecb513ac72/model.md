[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal way to transport goods from 10 specified source locations (S1–S10) to 20 specified demand locations (D1–D20) using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are exactly met, no source exceeds its supply, and the total transportation cost (cost per unit, summed over all units shipped) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are:
    - Sources: S = {S1, S2, ..., S10} (from expanded_sources.csv, all rows)
    - Destinations: D = {D1, D2, ..., D20} (from expanded_destinations.csv, all rows)
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source s to destination d. Type: GRB.INTEGER, t[s,d] ≥ 0.
    -   `f[s,d]` = Amount of cargo (units) shipped from source s to destination d. Type: GRB.CONTINUOUS, 0 ≤ f[s,d] ≤ 10 * t[s,d].
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from s to d: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand at each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (problem statement).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit from s to d) × (units shipped from s to d):  
    Minimize ∑_{s∈S} ∑_{d∈D} [cost[s,d] * f[s,d]]
7.  **Formulate Constraints:**
    -   **Supply Constraint:** For each source s, the total units shipped from s cannot exceed its supply:  
        ∑_{d∈D} f[s,d] ≤ supply_units[s]  for all s ∈ S
    -   **Demand Constraint:** For each destination d, the total units received must exactly meet its demand:  
        ∑_{s∈S} f[s,d] = demand_units[d]  for all d ∈ D
    -   **Truck Loading Constraint:** For each (s,d) pair, the amount shipped cannot exceed the total capacity of the dispatched trucks:  
        f[s,d] ≤ 10 * t[s,d]  for all s ∈ S, d ∈ D
    -   **Truck Integer Constraint:** For each (s,d), the number of trucks dispatched must be an integer:  
        t[s,d] ∈ {0, 1, 2, ...}
    -   **Non-negativity:** For all (s,d), f[s,d] ≥ 0
[Abstract Model Plan END]