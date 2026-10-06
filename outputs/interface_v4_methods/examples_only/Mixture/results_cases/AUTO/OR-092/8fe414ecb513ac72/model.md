[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal way to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum capacity of 10 units per trip, such that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit, not per truck) is minimized. The number of trucks dispatched per route must be integer, but each truck can be partially loaded.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for cargo allocation per truck.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t[s,d] \geq 0 \).
    -   `q[s,d]` = Quantity of cargo shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq q[s,d] \leq 10 \cdot t[s,d] \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from each source to each destination: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s \to d \)) × (quantity shipped \( q[s,d] \)).
7.  **Formulate Constraints:**
    -   **Supply Constraint:** For each source \( s \), the total quantity shipped from \( s \) to all destinations cannot exceed its supply:  
        \(\sum_{d \in D} q[s,d] \leq \text{expanded_sources.csv['supply_units'][s]}\)
    -   **Demand Constraint:** For each destination \( d \), the total quantity received from all sources must exactly meet its demand:  
        \(\sum_{s \in S} q[s,d] = \text{expanded_destinations.csv['demand_units'][d]}\)
    -   **Truck Capacity Constraint:** For each route \( (s,d) \), the quantity shipped cannot exceed the total capacity of the dispatched trucks:  
        \(q[s,d] \leq 10 \cdot t[s,d]\)
    -   **Truck Integer Constraint:** For each route \( (s,d) \), the number of trucks dispatched must be an integer:  
        \(t[s,d] \in \mathbb{Z}_{\geq 0}\)
    -   **Non-negativity:** For all \( (s,d) \), \(q[s,d] \geq 0\)
[Abstract Model Plan END]