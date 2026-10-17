[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from 10 source locations to 20 demand locations, using integer numbers of trucks (each with a maximum load of 10 units, but partial loading allowed), to meet all fixed demands at minimum total transportation cost. The cost is per unit shipped, and each source has a limited supply.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Linear Programming (MILP) transportation problem with integer variables for truck counts and continuous variables for cargo allocation per route.
3.  **Define Index Sets:** The primary indices are:
    - Sources: \( S = \{S1, S2, ..., S10\} \) (from expanded_sources.csv)
    - Destinations: \( D = \{D1, D2, ..., D20\} \) (from expanded_destinations.csv)
4.  **Define Decision Variables:**
    - \( t_{s,d} \) = Number of trucks dispatched from source \( s \) to destination \( d \). Type: GRB.INTEGER, \( t_{s,d} \geq 0 \).
    - \( x_{s,d} \) = Amount of cargo (units) shipped from source \( s \) to destination \( d \). Type: GRB.CONTINUOUS, \( 0 \leq x_{s,d} \leq 10 \cdot t_{s,d} \).
5.  **Identify Parameters (from Schema):**
    - Transportation cost per unit: from expanded_cost_matrix.csv, field [row: source_id, column: destination_id].
    - Source supply limits: from expanded_sources.csv, field 'supply_units' (by source_id).
    - Destination demand requirements: from expanded_destinations.csv, field 'demand_units' (by destination_id).
    - Truck capacity: fixed at 10 units per truck (from query).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all sources and destinations of (cost per unit) × (units shipped):  
       Minimize \( \sum_{s \in S} \sum_{d \in D} \text{cost}_{s,d} \cdot x_{s,d} \).
7.  **Formulate Constraints:**
    - **Supply Limit at Each Source:** For each source \( s \), the total units shipped from \( s \) cannot exceed its supply:  
      \( \sum_{d \in D} x_{s,d} \leq \text{supply\_units}_s \).
    - **Demand Satisfaction at Each Destination:** For each destination \( d \), the total units received must exactly meet its demand:  
      \( \sum_{s \in S} x_{s,d} = \text{demand\_units}_d \).
    - **Truck Loading Constraint:** For each route (source \( s \), destination \( d \)), the amount shipped cannot exceed the total capacity of the dispatched trucks:  
      \( x_{s,d} \leq 10 \cdot t_{s,d} \).
    - **Truck Integer Constraint:** For each route (source \( s \), destination \( d \)), the number of trucks dispatched must be an integer:  
      \( t_{s,d} \in \mathbb{Z}_{\geq 0} \).
    - **Cargo Nonnegativity:** For each route (source \( s \), destination \( d \)), the amount shipped must be nonnegative:  
      \( x_{s,d} \geq 0 \).
[Abstract Model Plan END]