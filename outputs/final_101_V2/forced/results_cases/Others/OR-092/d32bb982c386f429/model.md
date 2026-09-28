[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum capacity of 10 units per trip. The number of trucks dispatched on each route must be an integer, but each truck can be partially loaded. The objective is to meet all fixed demands at minimum total transportation cost, given per-unit route costs, source supply limits, and demand requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for partial truck loads.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{ij} \) = Number of trucks dispatched from source \( i \) to destination \( j \). Type: GRB.INTEGER (must be whole trucks, can be zero).
    -   \( q_{ij} \) = Quantity of cargo (units) shipped from source \( i \) to destination \( j \). Type: GRB.CONTINUOUS (can be any value between 0 and \( 10 \times t_{ij} \)).
5.  **Identify Parameters (from Schema):**
    -   Per-unit transportation cost for each route (\( c_{ij} \)): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source (\( s_i \)): from expanded_sources.csv, column 'supply_units'.
    -   Demand requirements for each destination (\( d_j \)): from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (problem statement).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize \( \sum_{i \in S} \sum_{j \in D} c_{ij} \cdot q_{ij} \), where \( c_{ij} \) is the per-unit cost for shipping from source \( i \) to destination \( j \).
7.  **Formulate Constraints:**
    -   **Supply Constraints:** For each source \( i \), the total quantity shipped out cannot exceed its supply: \( \sum_{j \in D} q_{ij} \leq s_i \).
    -   **Demand Constraints:** For each destination \( j \), the total quantity received must exactly meet its demand: \( \sum_{i \in S} q_{ij} = d_j \).
    -   **Truck Loading Constraints:** For each route \( (i, j) \), the quantity shipped cannot exceed the total capacity of the dispatched trucks: \( q_{ij} \leq 10 \cdot t_{ij} \).
    -   **Truck Integer Constraints:** For each route \( (i, j) \), \( t_{ij} \) must be an integer \( \geq 0 \).
    -   **Non-negativity:** For each route \( (i, j) \), \( q_{ij} \geq 0 \).
[Abstract Model Plan END]