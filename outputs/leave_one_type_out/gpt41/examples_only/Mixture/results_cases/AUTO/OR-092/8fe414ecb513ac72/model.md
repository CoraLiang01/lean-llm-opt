[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20), using trucks with a maximum capacity of 10 units per trip (integer number of trucks per route), such that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, not per truck) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for partial truck loading.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   `t[s, d]` = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t[s, d] \geq 0 \).
    -   `q[s, d]` = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq q[s, d] \leq 10 \cdot t[s, d] \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit for each route: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s \to d \)) × (units shipped on that route):  
        Minimize \( \sum_{s \in S} \sum_{d \in D} \text{cost}[s, d] \cdot q[s, d] \)
7.  **Formulate Constraints:**
    -   **Supply constraint (for each source):** The total units shipped from each source cannot exceed its available supply:  
        \( \sum_{d \in D} q[s, d] \leq \text{supply\_units}[s] \) for all \( s \in S \)
    -   **Demand constraint (for each destination):** The total units received at each destination must exactly meet its demand:  
        \( \sum_{s \in S} q[s, d] = \text{demand\_units}[d] \) for all \( d \in D \)
    -   **Truck loading constraint (for each route):** The amount shipped on each route cannot exceed the total capacity of the dispatched trucks:  
        \( q[s, d] \leq 10 \cdot t[s, d] \) for all \( s \in S, d \in D \)
    -   **Truck integer constraint:** The number of trucks dispatched on each route must be an integer:  
        \( t[s, d] \in \mathbb{Z}_{\geq 0} \) for all \( s \in S, d \in D \)
    -   **Non-negativity:**  
        \( q[s, d] \geq 0 \) for all \( s \in S, d \in D \)
[Abstract Model Plan END]