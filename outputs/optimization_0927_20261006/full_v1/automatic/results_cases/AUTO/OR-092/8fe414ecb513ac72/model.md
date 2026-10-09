[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 source locations to 20 demand locations, using integer numbers of trucks (each with up to 10 units, but allowing partial loads), to meet all fixed demands at minimum total transportation cost. The cost is per unit shipped, and each source has a limited supply.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER (must be integer, can be zero).
    -   \( q_{s,d} \) = Quantity of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS (can be fractional, \( 0 \leq q_{s,d} \leq 10 \cdot t_{s,d} \)).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (problem statement).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s \)–\( d \)) × (quantity shipped \( q_{s,d} \)).
7.  **Formulate Constraints:**
    -   Supply constraint: For each source \( s \), the total quantity shipped from \( s \) to all destinations cannot exceed its supply, i.e., \( \sum_{d \in D} q_{s,d} \leq \text{supply\_units}[s] \).
    -   Demand constraint: For each destination \( d \), the total quantity received from all sources must exactly meet its demand, i.e., \( \sum_{s \in S} q_{s,d} = \text{demand\_units}[d] \).
    -   Truck loading constraint: For each route \( (s,d) \), the quantity shipped cannot exceed the total capacity of the dispatched trucks, i.e., \( q_{s,d} \leq 10 \cdot t_{s,d} \).
    -   Truck integrality: For each route \( (s,d) \), \( t_{s,d} \) must be integer and \( \geq 0 \).
    -   Non-negativity: For each route \( (s,d) \), \( q_{s,d} \geq 0 \).
[Abstract Model Plan END]