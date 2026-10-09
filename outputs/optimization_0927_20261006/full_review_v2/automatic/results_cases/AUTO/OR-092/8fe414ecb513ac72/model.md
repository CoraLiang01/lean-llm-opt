[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 source locations to 20 demand locations using integer numbers of trucks (each with a maximum load of 10 units, but partial loading allowed), such that all supply and demand constraints are satisfied and the total transportation cost (cost per unit of cargo) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and continuous variables for cargo allocation.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER (must be integer, can be zero).
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS (can be any value between 0 and \( 10 \cdot t_{s,d} \)).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( s \) to destination \( d \): from expanded_cost_matrix.csv, columns D1–D20 for each source_id.
    -   Supply limit at each source \( s \): from expanded_sources.csv, column 'supply_units'.
    -   Demand requirement at each destination \( d \): from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit) × (units shipped):  
    \[
    \text{Minimize} \quad \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d}
    \]
7.  **Formulate Constraints:**
    -   **Truck Capacity Constraint:** For each source-destination pair, the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{s,d} \leq 10 \cdot t_{s,d} \quad \forall s \in S, d \in D
        \]
    -   **Supply Constraint:** For each source, the total amount shipped from that source cannot exceed its available supply:
        \[
        \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s \quad \forall s \in S
        \]
    -   **Demand Constraint:** For each destination, the total amount received must exactly meet its demand:
        \[
        \sum_{s \in S} x_{s,d} = \text{demand\_units}_d \quad \forall d \in D
        \]
    -   **Non-negativity and Integrality:**  
        \[
        x_{s,d} \geq 0 \quad \forall s \in S, d \in D
        \]
        \[
        t_{s,d} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S, d \in D
        \]
[Abstract Model Plan END]