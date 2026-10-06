[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum capacity of 10 units per trip, such that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit, not per truck) is minimized. The number of trucks dispatched per route must be integer, but truck loads can be partial (i.e., a truck may carry less than 10 units).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for cargo allocation per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( x_{s,d} \geq 0 \).
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from each source to each destination: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query, not CSV).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route) × (units shipped on that route):  
        Minimize \( \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d} \)
7.  **Formulate Constraints:**
    -   **Supply constraint (for each source):** The total units shipped from each source cannot exceed its available supply:  
        \( \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s \) for all \( s \in S \)
    -   **Demand constraint (for each destination):** The total units received at each destination must exactly meet its demand:  
        \( \sum_{s \in S} x_{s,d} = \text{demand\_units}_d \) for all \( d \in D \)
    -   **Truck loading constraint (for each route):** The amount shipped on each route cannot exceed the total capacity of the dispatched trucks:  
        \( x_{s,d} \leq 10 \cdot t_{s,d} \) for all \( s \in S, d \in D \)
    -   **Truck usage constraint:** If any cargo is shipped on a route, at least one truck must be dispatched; the number of trucks per route must be integer and non-negative.
    -   **Partial loading allowed:** \( x_{s,d} \) can be any value between 0 and \( 10 \cdot t_{s,d} \); \( t_{s,d} \) must be integer.
    -   **Non-negativity:** \( x_{s,d} \geq 0 \), \( t_{s,d} \geq 0 \), and \( t_{s,d} \) integer for all \( s, d \).
[Abstract Model Plan END]