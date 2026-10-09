[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 source locations to 20 demand locations, using integer numbers of trucks (each with a maximum load of 10 units, but partial loading allowed), to meet all fixed demands at minimum total transportation cost. The cost is per unit of cargo, and each source has a limited supply.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    - \( t_{s,d} \): Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER (must be integer, can be zero).
    - \( x_{s,d} \): Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS (can be any value between 0 and \( 10 \cdot t_{s,d} \)).
5.  **Identify Parameters (from Schema):**
    - Transportation cost per unit from source \( s \) to destination \( d \): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    - Supply limit at each source \( s \): from expanded_sources.csv, column supply_units.
    - Demand at each destination \( d \): from expanded_destinations.csv, column demand_units.
    - Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s,d \)) × (units shipped \( x_{s,d} \)).
7.  **Formulate Constraints:**
    - **Supply constraint:** For each source \( s \), the total units shipped from \( s \) to all destinations cannot exceed its supply: \( \sum_{d} x_{s,d} \leq \text{expanded_sources[s]['supply_units']} \).
    - **Demand constraint:** For each destination \( d \), the total units received from all sources must exactly meet its demand: \( \sum_{s} x_{s,d} = \text{expanded_destinations[d]['demand_units']} \).
    - **Truck loading constraint:** For each route \( (s,d) \), the amount shipped cannot exceed the total capacity of the dispatched trucks: \( x_{s,d} \leq 10 \cdot t_{s,d} \).
    - **Truck integer constraint:** For each route \( (s,d) \), \( t_{s,d} \) must be an integer ≥ 0.
    - **Cargo non-negativity:** For each route \( (s,d) \), \( x_{s,d} \geq 0 \).
[Abstract Model Plan END]