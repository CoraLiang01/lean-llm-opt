[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan to ship goods from 10 specified sources (S1–S10) to 20 specified demand locations (D1–D20), using integer numbers of trucks per route (each truck can carry up to 10 units, but may be partially loaded), so that all demands are exactly met, no source exceeds its supply, and the total transportation cost (cost per unit shipped, summed over all routes) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and partial truck loading allowed.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( i \in \{S1, S2, ..., S10\} \)
    - Destinations: \( j \in \{D1, D2, ..., D20\} \)
4.  **Define Decision Variables:**
    -   `t[i,j]` = Number of trucks dispatched from source \(i\) to destination \(j\). Type: GRB.INTEGER (must be a non-negative integer).
    -   `x[i,j]` = Amount of cargo (units) shipped from source \(i\) to destination \(j\). Type: GRB.CONTINUOUS (bounded between 0 and \(10 \times t[i,j]\)).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit for each route: from `expanded_cost_matrix.csv`, columns `D1`–`D20` for each `source_id`.
    -   Supply limits for each source: from `expanded_sources.csv`, column `supply_units` for each `source_id`.
    -   Demand requirements for each destination: from `expanded_destinations.csv`, column `demand_units` for each `destination_id`.
    -   Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit on route \(i,j\)) × (units shipped on route \(i,j\)), or:
    \[
    \text{Minimize} \quad \sum_{i \in \text{Sources}} \sum_{j \in \text{Destinations}} \text{cost}[i,j] \times x[i,j]
    \]
7.  **Formulate Constraints:**
    -   **Supply Constraints:** For each source \(i\), the total units shipped from \(i\) cannot exceed its supply:
        \[
        \sum_{j} x[i,j] \leq \text{supply\_units}[i]
        \]
    -   **Demand Constraints:** For each destination \(j\), the total units received must exactly meet its demand:
        \[
        \sum_{i} x[i,j] = \text{demand\_units}[j]
        \]
    -   **Truck Loading Constraints:** For each route \((i,j)\), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x[i,j] \leq 10 \times t[i,j]
        \]
    -   **Truck Integer Constraints:** For each route \((i,j)\), the number of trucks dispatched must be a non-negative integer:
        \[
        t[i,j] \in \mathbb{Z}_{\geq 0}
        \]
    -   **Non-negativity of Shipments:** For each route \((i,j)\), the amount shipped must be non-negative:
        \[
        x[i,j] \geq 0
        \]
[Abstract Model Plan END]