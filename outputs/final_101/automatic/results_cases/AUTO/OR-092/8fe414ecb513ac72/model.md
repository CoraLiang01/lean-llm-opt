[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20), using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are exactly met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for cargo allocation per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   `t[s,d]` = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER (must be whole trucks, can be zero).
    -   `x[s,d]` = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS (can be fractional, up to 10 per truck, can be zero).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit for each route: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query description).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s,d \)) × (units shipped on route \( s,d \)), or:
    - Minimize \( \sum_{s \in S} \sum_{d \in D} \text{cost}[s,d] \times x[s,d] \)
7.  **Formulate Constraints:**
    -   **Truck Loading Constraint:** For each route, the total cargo shipped cannot exceed the total capacity of the dispatched trucks:
        - For all \( s \in S, d \in D \): \( x[s,d] \leq 10 \times t[s,d] \)
    -   **Supply Constraint:** For each source, the total cargo shipped out cannot exceed its available supply:
        - For all \( s \in S \): \( \sum_{d \in D} x[s,d] \leq \text{supply\_units}[s] \)
    -   **Demand Satisfaction Constraint:** For each destination, the total cargo received must exactly meet its demand:
        - For all \( d \in D \): \( \sum_{s \in S} x[s,d] = \text{demand\_units}[d] \)
    -   **Non-negativity and Integrality:**
        - For all \( s,d \): \( x[s,d] \geq 0 \) (continuous)
        - For all \( s,d \): \( t[s,d] \geq 0 \) and integer
[Abstract Model Plan END]