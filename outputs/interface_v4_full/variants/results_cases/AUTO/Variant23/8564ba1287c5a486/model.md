[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipping plan from depots to markets, where each route incurs a fixed activation cost if used and a variable cost per unit shipped. Each depot has a supply capacity, each market has a demand, and shipments on a route are only allowed if the route is activated. The model must ensure all market demands are met, depot capacities are not exceeded, and route activation is properly linked to shipments.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Depots (from depot_capacity.csv): D1, D2, D3, D4
    - Markets (from market_demand.csv): M1, M2, M3, M4, M5
    - Routes: All combinations of (Depot, Market)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Quantity shipped from depot i to market j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from depot i to market j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping cost per unit (`c[i,j]`): from route_variable_costs.csv, columns M1–M5 for each depot.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns M1–M5 for each depot.
    -   Depot supply capacity (`S[i]`): from depot_capacity.csv, column 'SupplyCapacity'.
    -   Market demand (`D[j]`): from market_demand.csv, column 'Demand'.
    -   Big-M for linking (`M[i,j]`): computed as min(S[i], D[j]) for each route (i,j).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable shipping cost per unit * quantity shipped) plus (fixed activation cost * route activation indicator):
        Minimize sum_{i in Depots, j in Markets} [ c[i,j] * x[i,j] + f[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Market Demand Satisfaction): For each market j, the total shipments received from all depots must meet its demand exactly:
            sum_{i in Depots} x[i,j] = D[j]   for all j in Markets
    -   Constraint 2 (Depot Supply Limit): For each depot i, the total shipments sent to all markets cannot exceed its supply capacity:
            sum_{j in Markets} x[i,j] <= S[i]   for all i in Depots
    -   Constraint 3 (Route Activation Linking): For each route (i,j), shipments can only occur if the route is activated, and cannot exceed the minimum of depot capacity and market demand:
            x[i,j] <= M[i,j] * y[i,j]   for all i in Depots, j in Markets
    -   Constraint 4 (Nonnegativity): x[i,j] >= 0   for all i in Depots, j in Markets
    -   Constraint 5 (Binary): y[i,j] in {0,1}   for all i in Depots, j in Markets
[Abstract Model Plan END]