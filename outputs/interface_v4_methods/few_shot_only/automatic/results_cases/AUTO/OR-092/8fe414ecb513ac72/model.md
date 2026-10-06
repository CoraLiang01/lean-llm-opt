[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20), using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \)
    - Destinations: \( D = \{D1, D2, ..., D20\} \)
4.  **Define Decision Variables:**
    -   \( t_{ij} \) = Number of trucks dispatched from source \( i \) to destination \( j \). Type: GRB.INTEGER (must be integer, can be zero).
    -   \( x_{ij} \) = Amount of cargo (units) shipped from source \( i \) to destination \( j \). Type: GRB.CONTINUOUS (can be fractional, \( 0 \leq x_{ij} \leq 10 \cdot t_{ij} \)).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( i \) to destination \( j \): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand at each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( i \)-\( j \)) × (units shipped on route \( i \)-\( j \)), or:
    \[
    \text{Minimize} \quad \sum_{i \in S} \sum_{j \in D} \text{cost}_{ij} \cdot x_{ij}
    \]
7.  **Formulate Constraints:**
    -   **Supply constraint (for each source):** The total units shipped from each source cannot exceed its available supply.
        \[
        \sum_{j \in D} x_{ij} \leq \text{supply\_units}_i \quad \forall i \in S
        \]
    -   **Demand constraint (for each destination):** The total units received at each destination must exactly meet its demand.
        \[
        \sum_{i \in S} x_{ij} = \text{demand\_units}_j \quad \forall j \in D
        \]
    -   **Truck loading constraint (for each route):** The amount shipped on each route cannot exceed the total capacity of the dispatched trucks.
        \[
        x_{ij} \leq 10 \cdot t_{ij} \quad \forall i \in S, \forall j \in D
        \]
    -   **Non-negativity and integrality:**
        \[
        x_{ij} \geq 0 \quad \forall i, j; \quad t_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
        \]
[Abstract Model Plan END]