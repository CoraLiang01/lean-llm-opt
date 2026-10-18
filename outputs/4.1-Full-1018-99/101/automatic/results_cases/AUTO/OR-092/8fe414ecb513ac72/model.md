[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 source locations to 20 demand locations, using integer numbers of trucks (each with a maximum load of 10 units, but partial loading allowed), to meet all fixed demands at minimum total transportation cost. The cost is per unit of cargo, not per truck. Each source has a limited supply, and each destination has a fixed demand.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for the number of trucks dispatched per route and continuous variables for the actual cargo shipped.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    - \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    - \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq x_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    - Transportation cost per unit from source \( s \) to destination \( d \): from expanded_cost_matrix.csv, columns ['D1', ..., 'D20'] for each 'source_id'.
    - Supply limit at each source \( s \): from expanded_sources.csv, column 'supply_units'.
    - Demand at each destination \( d \): from expanded_destinations.csv, column 'demand_units'.
    - Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s,d \)) × (units shipped \( x_{s,d} \)).
7.  **Formulate Constraints:**
    - Supply constraint for each source: The total units shipped from each source across all destinations cannot exceed its supply limit.
    - Demand constraint for each destination: The total units received at each destination from all sources must exactly meet its demand.
    - Truck loading constraint for each route: The amount shipped on each route cannot exceed 10 times the number of trucks dispatched on that route (\( x_{s,d} \leq 10 \cdot t_{s,d} \)), and \( t_{s,d} \) must be integer and non-negative.
    - Non-negativity: All \( x_{s,d} \geq 0 \), all \( t_{s,d} \geq 0 \) and integer.
[Abstract Model Plan END]