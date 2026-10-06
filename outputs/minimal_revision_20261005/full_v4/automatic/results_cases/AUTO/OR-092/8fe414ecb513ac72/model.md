[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20), using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are exactly met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( i \in \{\text{S1}, \text{S2}, ..., \text{S10}\} \)
    - Destinations: \( j \in \{\text{D1}, \text{D2}, ..., \text{D20}\} \)
4.  **Define Decision Variables:**
    -   \( t_{i,j} \) = Number of trucks dispatched from source \( i \) to destination \( j \). Type: GRB.INTEGER (must be integer, can be zero).
    -   \( x_{i,j} \) = Amount of cargo (units) shipped from source \( i \) to destination \( j \). Type: GRB.CONTINUOUS (can be fractional, \( 0 \leq x_{i,j} \leq 10 \cdot t_{i,j} \)).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( i \) to destination \( j \): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand at each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \( i,j \)) × (units shipped on route \( i,j \)), or:
    \[
    \text{Minimize} \quad \sum_{i \in \text{Sources}} \sum_{j \in \text{Destinations}} \text{cost}_{i,j} \cdot x_{i,j}
    \]
7.  **Formulate Constraints:**
    -   **Supply Constraints:** For each source \( i \), the total units shipped from \( i \) cannot exceed its supply:
        \[
        \sum_{j} x_{i,j} \leq \text{supply\_units}_i
        \]
    -   **Demand Constraints:** For each destination \( j \), the total units received must exactly meet its demand:
        \[
        \sum_{i} x_{i,j} = \text{demand\_units}_j
        \]
    -   **Truck Loading Constraints:** For each route \( (i,j) \), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{i,j} \leq 10 \cdot t_{i,j}
        \]
        (where \( t_{i,j} \) is integer, and \( x_{i,j} \) can be any value between 0 and \( 10 \cdot t_{i,j} \))
    -   **Non-negativity and Integrality:**
        \[
        t_{i,j} \geq 0, \quad t_{i,j} \in \mathbb{Z}
        \]
        \[
        x_{i,j} \geq 0
        \]
[Abstract Model Plan END]