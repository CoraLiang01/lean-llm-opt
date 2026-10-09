[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost way to ship goods from a set of depots to a set of markets, considering both per-unit shipping costs and fixed activation costs for each possible depot-market route. Shipments on a route are only allowed if the route is activated. The model must ensure all market demands are met, depot capacities are not exceeded, and route activation is properly linked to shipment quantities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Depots (from depot_capacity.csv) and Markets (from market_demand.csv). Each route is defined by a (Depot, Market) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity shipped from depot i to market j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from depot i to market j is activated, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping cost per unit (`c[i,j]`): from route_variable_costs.csv, columns 'Depot', 'M1', ..., 'M5'.
    -   Fixed activation cost per route (`f[i,j]`): from route_fixed_costs.csv, columns 'Depot', 'M1', ..., 'M5'.
    -   Depot supply capacity (`S[i]`): from depot_capacity.csv, columns 'Depot', 'SupplyCapacity'.
    -   Market demand (`D[j]`): from market_demand.csv, columns 'Market', 'Demand'.
    -   Maximum possible shipment per route (`M[i,j]`): computed as min(S[i], D[j]) for each (i,j) pair.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable shipping cost per unit * shipment quantity) plus (fixed activation cost * route activation variable):  
        Minimize sum_{i,j} [ c[i,j] * x[i,j] + f[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Market Demand Satisfaction): For each market j, the total shipments received from all depots must equal the market's demand:  
            sum_{i} x[i,j] = D[j]  for all j
    -   Constraint 2 (Depot Supply Capacity): For each depot i, the total shipments sent to all markets cannot exceed the depot's capacity:  
            sum_{j} x[i,j] <= S[i]  for all i
    -   Constraint 3 (Route Activation Linking): For each route (i,j), shipments are only allowed if the route is activated, and cannot exceed the maximum possible shipment:  
            x[i,j] <= M[i,j] * y[i,j]  for all (i,j)
    -   Constraint 4 (Nonnegativity):  
            x[i,j] >= 0  for all (i,j)
    -   Constraint 5 (Binary Route Activation):  
            y[i,j] in {0,1}  for all (i,j)
[Abstract Model Plan END]