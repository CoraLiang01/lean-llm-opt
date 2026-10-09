[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how to transport goods from 10 source locations to 20 demand locations using trucks (each with a maximum load of 10 units, but allowing partial loads), such that all demands are met, no source exceeds its supply, only integer numbers of trucks are dispatched per route, and the total transportation cost (cost per unit of cargo) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( x_{s,d} \geq 0 \).
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source to destination: from expanded_cost_matrix.csv, columns ['source_id', 'D1', ..., 'D20'].
    -   Supply limits per source: from expanded_sources.csv, columns ['source_id', 'supply_units'].
    -   Demand requirements per destination: from expanded_destinations.csv, columns ['destination_id', 'demand_units'].
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize \( \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d} \), where \(\text{cost}_{s,d}\) is the per-unit cost from expanded_cost_matrix.csv.
7.  **Formulate Constraints:**
    -   Supply constraint: For each source \( s \), \( \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s \) (from expanded_sources.csv).
    -   Demand constraint: For each destination \( d \), \( \sum_{s \in S} x_{s,d} = \text{demand\_units}_d \) (from expanded_destinations.csv).
    -   Truck loading constraint: For each route \( (s,d) \), \( x_{s,d} \leq 10 \cdot t_{s,d} \) (truck can carry up to 10 units; partial loading allowed, but number of trucks must be integer).
    -   Truck integrality: For each route \( (s,d) \), \( t_{s,d} \) is integer and \( t_{s,d} \geq 0 \).
    -   Non-negativity: For each route \( (s,d) \), \( x_{s,d} \geq 0 \).
[Abstract Model Plan END]