[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations to 20 demand locations using integer numbers of trucks (each with a maximum load of 10 units, but partial loading allowed), such that all supply and demand constraints are satisfied and the total transportation cost (cost per unit of cargo) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{i,j} \) = Number of trucks dispatched from source \( i \) to destination \( j \). Type: GRB.INTEGER (must be integer, can be zero).
    -   \( x_{i,j} \) = Amount of cargo (units) shipped from source \( i \) to destination \( j \). Type: GRB.CONTINUOUS (can be fractional, between 0 and \( 10 \cdot t_{i,j} \)).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( i \) to destination \( j \): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source \( i \): from expanded_sources.csv, column 'supply_units'.
    -   Demand requirement at each destination \( j \): from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (from query description).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all source-destination pairs of (cost per unit) × (units shipped):  
    \[
    \text{Minimize} \quad \sum_{i \in S} \sum_{j \in D} \text{cost}_{i,j} \cdot x_{i,j}
    \]
7.  **Formulate Constraints:**
    -   **Supply Constraints:** For each source \( i \), the total units shipped from \( i \) cannot exceed its supply:
        \[
        \sum_{j \in D} x_{i,j} \leq \text{supply\_units}_i
        \]
    -   **Demand Constraints:** For each destination \( j \), the total units received must exactly meet its demand:
        \[
        \sum_{i \in S} x_{i,j} = \text{demand\_units}_j
        \]
    -   **Truck Capacity Constraints:** For each route \( (i, j) \), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{i,j} \leq 10 \cdot t_{i,j}
        \]
    -   **Truck Integer Constraints:** For each route \( (i, j) \), the number of trucks dispatched must be an integer:
        \[
        t_{i,j} \in \mathbb{Z}_{\geq 0}
        \]
    -   **Cargo Non-negativity:** For each route \( (i, j) \), the amount shipped must be non-negative:
        \[
        x_{i,j} \geq 0
        \]
[Abstract Model Plan END]