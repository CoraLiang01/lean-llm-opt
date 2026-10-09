[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which plants to open and how much each open plant should ship to each customer, in order to minimize the total cost (fixed plant opening costs plus variable transportation costs), while meeting all customer demands and not exceeding plant capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge transportation) problem.
3.  **Define Index Sets:** The primary indices are Plants (F1–F15, from 'plant' in cost.csv) and Customers (C1–C15, from 'customer' in demand.csv).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Amount shipped from plant i to customer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i]` = 1 if plant i is opened (built), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each plant: 'fixed_cost' column in cost.csv.
    -   Plant capacity: 'capacity' column in cost.csv.
    -   Per-unit transport cost from plant i to customer j: columns 'C1'–'C15' in cost.csv.
    -   Customer demand: 'demand' column in demand.csv.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of fixed opening costs for opened plants plus the sum of transportation costs for all shipments:  
        sum over i (fixed_cost[i] * y[i]) + sum over i,j (transport_cost[i,j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each customer j, the total amount received from all plants must equal the demand for customer j (sum over i of x[i,j] = demand[j]).
    -   Plant Capacity: For each plant i, the total amount shipped from plant i to all customers cannot exceed its capacity if the plant is opened (sum over j of x[i,j] ≤ capacity[i] * y[i]).
    -   Linking and Nonnegativity: x[i,j] ≥ 0 for all i, j; y[i] ∈ {0,1} for all i.
[Abstract Model Plan END]