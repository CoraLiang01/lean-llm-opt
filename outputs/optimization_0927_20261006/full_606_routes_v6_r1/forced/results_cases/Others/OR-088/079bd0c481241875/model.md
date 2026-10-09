[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which plants to open and how much each open plant should ship to each customer, in order to minimize the total cost (fixed opening costs plus variable transportation costs), while meeting all customer demands and not exceeding plant capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge transportation) problem.
3.  **Define Index Sets:** The primary indices are Plants (F1–F15) and Customers (C1–C15).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Amount shipped from plant i to customer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i]` = 1 if plant i is opened (built), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed opening costs: from 'fixed_cost' column in cost.csv (per plant).
    -   Plant capacities: from 'capacity' column in cost.csv (per plant).
    -   Per-unit transport costs: from columns 'C1'–'C15' in cost.csv (per plant-customer pair).
    -   Customer demands: from 'demand' column in demand.csv (per customer).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of fixed opening costs for opened plants plus the sum of transportation costs for all shipments:  
        sum over i of (fixed_cost[i] * y[i]) + sum over i,j of (transport_cost[i,j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each customer j, the total amount shipped to j from all plants must equal demand[j]:  
        sum over i of x[i,j] = demand[j] for all j.
    -   Plant Capacity: For each plant i, the total amount shipped from i to all customers cannot exceed its capacity if opened:  
        sum over j of x[i,j] ≤ capacity[i] * y[i] for all i.
    -   Linking and Nonnegativity:  
        x[i,j] ≥ 0 for all i, j;  
        y[i] ∈ {0,1} for all i.
[Abstract Model Plan END]