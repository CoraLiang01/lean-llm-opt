[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20), using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq x_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit for each route: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query description).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s \to d \)) × (units shipped on that route):  
    \[
    \text{Minimize} \quad \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d}
    \]
7.  **Formulate Constraints:**
    -   **Supply constraints:** For each source \( s \), the total amount shipped from \( s \) cannot exceed its supply:
        \[
        \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s
        \]
    -   **Demand constraints:** For each destination \( d \), the total amount received must exactly meet its demand:
        \[
        \sum_{s \in S} x_{s,d} = \text{demand\_units}_d
        \]
    -   **Truck loading constraints:** For each route \( (s,d) \), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{s,d} \leq 10 \cdot t_{s,d}
        \]
    -   **Truck integrality:** For each route \( (s,d) \), the number of trucks dispatched must be an integer:
        \[
        t_{s,d} \in \mathbb{Z}_{\geq 0}
        \]
    -   **Non-negativity:** For each route \( (s,d) \), the amount shipped must be non-negative:
        \[
        x_{s,d} \geq 0
        \]
[Abstract Model Plan END]