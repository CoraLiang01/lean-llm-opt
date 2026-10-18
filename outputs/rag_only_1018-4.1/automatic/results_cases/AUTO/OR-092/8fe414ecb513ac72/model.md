[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal way to transport goods from 10 source locations to 20 demand locations using trucks with a maximum capacity of 10 units, such that all demands are met, no source exceeds its supply, only whole trucks are dispatched per route (integer number of trucks), and the total transportation cost (cost per unit of cargo) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Linear Programming (MIP) transportation problem with integer variables for the number of trucks dispatched per route and continuous variables for the amount of cargo shipped.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq x_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source to destination: from expanded_cost_matrix.csv, columns ['source_id', 'D1', ..., 'D20'].
    -   Supply limits per source: from expanded_sources.csv, column 'supply_units'.
    -   Demand requirements per destination: from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all sources and destinations of (cost per unit) × (units shipped):  
        \(\text{Minimize} \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d}\)
7.  **Formulate Constraints:**
    -   **Supply Constraint:** For each source \( s \), the total units shipped from \( s \) cannot exceed its supply:  
        \(\sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s\)
    -   **Demand Constraint:** For each destination \( d \), the total units received must exactly meet its demand:  
        \(\sum_{s \in S} x_{s,d} = \text{demand\_units}_d\)
    -   **Truck Loading Constraint:** For each route (source \( s \), destination \( d \)), the amount shipped cannot exceed the total capacity of the dispatched trucks:  
        \(x_{s,d} \leq 10 \cdot t_{s,d}\)
    -   **Truck Integer Constraint:** For each route (source \( s \), destination \( d \)), the number of trucks dispatched must be an integer:  
        \(t_{s,d} \in \mathbb{Z}_{\geq 0}\)
    -   **Non-negativity:** For each route (source \( s \), destination \( d \)), the amount shipped must be non-negative:  
        \(x_{s,d} \geq 0\)
[Abstract Model Plan END]