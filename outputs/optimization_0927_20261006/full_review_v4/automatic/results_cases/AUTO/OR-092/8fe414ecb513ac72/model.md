[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 source locations to 20 demand locations, using integer numbers of trucks (each with a maximum load of 10 units, but partial loading allowed), to meet all fixed demands at minimum total transportation cost. The cost is per unit of cargo, not per truck. Each source has a limited supply, and each destination has a fixed demand.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for the number of trucks dispatched per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    -   \( f_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq f_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( s \) to destination \( d \): from expanded_cost_matrix.csv, columns D1–D20.
    -   Supply limit at each source \( s \): from expanded_sources.csv, column 'supply_units'.
    -   Demand at each destination \( d \): from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s,d \)) × (units shipped \( f_{s,d} \)).
7.  **Formulate Constraints:**
    -   **Supply Constraint:** For each source \( s \), the total units shipped from \( s \) to all destinations cannot exceed its supply: \( \sum_{d \in D} f_{s,d} \leq \text{expanded_sources['supply_units'][s]} \).
    -   **Demand Constraint:** For each destination \( d \), the total units received from all sources must exactly meet its demand: \( \sum_{s \in S} f_{s,d} = \text{expanded_destinations['demand_units'][d]} \).
    -   **Truck Loading Constraint:** For each route \( (s,d) \), the amount shipped cannot exceed the total capacity of the dispatched trucks: \( f_{s,d} \leq 10 \cdot t_{s,d} \).
    -   **Truck Integer Constraint:** For each route \( (s,d) \), the number of trucks dispatched \( t_{s,d} \) must be an integer \( \geq 0 \).
    -   **Partial Loading Allowed:** For each route \( (s,d) \), \( f_{s,d} \) can be any value in \( [0, 10 \cdot t_{s,d}] \), allowing the last truck to be partially loaded.
[Abstract Model Plan END]