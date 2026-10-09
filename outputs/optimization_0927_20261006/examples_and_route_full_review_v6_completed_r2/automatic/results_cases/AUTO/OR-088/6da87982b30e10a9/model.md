[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which plants to build and how much each built plant should supply to each customer, in order to minimize the total cost (fixed plant opening costs plus variable transportation costs), while meeting all customer demands and not exceeding plant capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge transportation) problem.
3.  **Define Index Sets:** The primary indices are Plants (F1–F15, from the 'plant' column in cost.csv) and Customers (C1–C15, from the 'customer' column in demand.csv).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Amount shipped from plant i to customer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i]` = 1 if plant i is built (opened), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each plant: 'fixed_cost' column in cost.csv.
    -   Plant capacity: 'capacity' column in cost.csv.
    -   Per-unit transport cost from plant i to customer j: columns 'C1'–'C15' in cost.csv.
    -   Customer demand: 'demand' column in demand.csv.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of fixed opening costs for built plants plus the sum of transportation costs for all shipments:  
        sum over i of (fixed_cost[i] * y[i]) + sum over i,j of (transport_cost[i,j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each customer j, the total amount received from all plants must equal the customer's demand (sum over i of x[i,j] = demand[j]).
    -   Plant Capacity: For each plant i, the total amount shipped from that plant to all customers cannot exceed its capacity if the plant is built (sum over j of x[i,j] ≤ capacity[i] * y[i]).
    -   Linking: Shipments from a plant are only allowed if the plant is built (enforced by the capacity constraint above).
    -   Nonnegativity: All shipment variables x[i,j] ≥ 0.
    -   Binary: All plant opening variables y[i] ∈ {0,1}.
[Abstract Model Plan END]