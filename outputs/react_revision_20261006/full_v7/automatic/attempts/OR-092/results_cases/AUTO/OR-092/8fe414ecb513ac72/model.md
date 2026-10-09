[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal way to transport goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using integer numbers of trucks (each with a maximum load of 10 units, but partial loading is allowed), so that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit of cargo, varying by route) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq x_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source to destination: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query description).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit for route) × (units shipped on that route):  
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
    -   **Truck Loading Constraints:** For each route (source \( s \), destination \( d \)), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{s,d} \leq 10 \cdot t_{s,d}
        \]
        where \( t_{s,d} \) is integer and \( x_{s,d} \) is continuous.
    -   **Non-negativity and Integrality:**  
        \[
        t_{s,d} \in \mathbb{Z}_{\geq 0}, \quad x_{s,d} \geq 0
        \]
[Abstract Model Plan END]