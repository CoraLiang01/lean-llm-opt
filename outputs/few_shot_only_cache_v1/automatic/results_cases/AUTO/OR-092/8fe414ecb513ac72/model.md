[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 source locations to 20 demand locations, using integer numbers of trucks (each with up to 10 units of cargo, but partial loading allowed), to meet all fixed demands at minimum total transportation cost. The cost is per unit of cargo, and each source has a limited supply.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck trips and partial truck loading.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    -   \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER (must be integer, can be zero).
    -   \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS (can be any value in [0, 10 * \( t_{s,d} \)]).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from source \( s \) to destination \( d \): from expanded_cost_matrix.csv, columns D1–D20.
    -   Supply limit at each source \( s \): from expanded_sources.csv, column 'supply_units'.
    -   Demand at each destination \( d \): from expanded_destinations.csv, column 'demand_units'.
    -   Truck capacity: 10 units (fixed constant).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all source-destination pairs of (cost per unit) × (units shipped):  
        \[
        \text{Minimize} \quad \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d}
        \]
7.  **Formulate Constraints:**
    -   **Supply Limit at Each Source:** For each source \( s \), the total units shipped from \( s \) cannot exceed its supply:
        \[
        \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s
        \]
    -   **Demand Satisfaction at Each Destination:** For each destination \( d \), the total units received must exactly meet its demand:
        \[
        \sum_{s \in S} x_{s,d} = \text{demand\_units}_d
        \]
    -   **Truck Capacity Linking:** For each source-destination pair \((s,d)\), the amount shipped cannot exceed the total capacity of the dispatched trucks:
        \[
        x_{s,d} \leq 10 \cdot t_{s,d}
        \]
    -   **Non-negativity and Integrality:** For all \( s, d \):
        \[
        x_{s,d} \geq 0 \quad (\text{continuous}), \quad t_{s,d} \geq 0 \quad (\text{integer})
        \]
[Abstract Model Plan END]