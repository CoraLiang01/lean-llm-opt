[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations (S1–S10) to 20 demand locations (D1–D20) using trucks with a maximum load of 10 units per trip. The number of trucks dispatched on each route must be an integer, but each truck can be partially loaded. The objective is to meet all fixed demands at minimum total transportation cost, given per-unit route costs, source supply limits, and demand requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for the number of trucks per route and continuous variables for the amount shipped per route (with linking constraints).
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( i \in \{\text{S1}, \text{S2}, ..., \text{S10}\} \)
    - Destinations: \( j \in \{\text{D1}, \text{D2}, ..., \text{D20}\} \)
4.  **Define Decision Variables:**
    -   \( x_{i,j} \) = Amount of cargo (units) shipped from source \( i \) to destination \( j \). Type: GRB.CONTINUOUS, \( x_{i,j} \geq 0 \).
    -   \( t_{i,j} \) = Number of trucks dispatched from source \( i \) to destination \( j \). Type: GRB.INTEGER, \( t_{i,j} \geq 0 \).
5.  **Identify Parameters (from Schema):**
    -   Per-unit transportation cost for each route: from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limits for each source: from expanded_sources.csv, column 'supply_units' for each source_id.
    -   Demand requirements for each destination: from expanded_destinations.csv, column 'demand_units' for each destination_id.
    -   Truck capacity: fixed at 10 units per truck (from query, not CSV).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (per-unit cost) × (units shipped):  
    \[
    \text{Minimize} \quad \sum_{i \in \text{Sources}} \sum_{j \in \text{Destinations}} \text{cost}_{i,j} \cdot x_{i,j}
    \]
7.  **Formulate Constraints:**
    -   **Supply Constraints:** For each source \( i \), the total amount shipped from \( i \) cannot exceed its supply:
        \[
        \sum_{j} x_{i,j} \leq \text{supply\_units}_i
        \]
    -   **Demand Constraints:** For each destination \( j \), the total amount received must exactly meet its demand:
        \[
        \sum_{i} x_{i,j} = \text{demand\_units}_j
        \]
    -   **Truck Capacity Constraints:** For each route \( (i,j) \), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{i,j} \leq 10 \cdot t_{i,j}
        \]
    -   **Truck Integer Constraints:** For each route \( (i,j) \), the number of trucks dispatched must be an integer:
        \[
        t_{i,j} \in \mathbb{Z}_{\geq 0}
        \]
    -   **Non-negativity:** For all \( (i,j) \), \( x_{i,j} \geq 0 \).
[Abstract Model Plan END]