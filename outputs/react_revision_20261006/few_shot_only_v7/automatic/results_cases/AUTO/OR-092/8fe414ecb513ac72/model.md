[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 specified source locations (S1–S10) to 20 specified demand locations (D1–D20), using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are exactly met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for per-route shipment quantities.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv, all rows)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv, all rows)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq x_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( s \) to destination \( d \): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source \( s \): from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand at each destination \( d \): from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( s \to d \)) × (units shipped on that route):  
    \[
    \text{Minimize} \quad \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d}
    \]
7.  **Formulate Constraints:**
    -   **Supply Constraints:** For each source \( s \), the total units shipped from \( s \) cannot exceed its supply:
        \[
        \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s
        \]
    -   **Demand Constraints:** For each destination \( d \), the total units received must exactly meet its demand:
        \[
        \sum_{s \in S} x_{s,d} = \text{demand\_units}_d
        \]
    -   **Truck Loading Constraints:** For each route \( (s,d) \), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{s,d} \leq 10 \cdot t_{s,d}
        \]
        (where \( t_{s,d} \) is integer, \( x_{s,d} \) is continuous and nonnegative)
    -   **Non-negativity and Integrality:** All \( x_{s,d} \geq 0 \), all \( t_{s,d} \geq 0 \) and integer.
[Abstract Model Plan END]