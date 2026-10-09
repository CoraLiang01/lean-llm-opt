[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipping plan from depots to markets, where each route incurs a fixed activation cost if used and a variable cost per unit shipped. The plan must satisfy all market demands, not exceed depot capacities, and only allow shipments on activated routes.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Depots (from depot_capacity.csv) and Markets (from market_demand.csv). Each route is defined by a (Depot, Market) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity shipped from depot i to market j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from depot i to market j is activated, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping cost per unit (`c[i,j]`): from route_variable_costs.csv, columns 'Depot', 'M1', ..., 'M5'.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns 'Depot', 'M1', ..., 'M5'.
    -   Depot supply capacity (`S[i]`): from depot_capacity.csv, columns 'Depot', 'SupplyCapacity'.
    -   Market demand (`D[j]`): from market_demand.csv, columns 'Market', 'Demand'.
    -   Maximum possible shipment on route (`M[i,j]`): computed as min(S[i], D[j]) for each (i,j) pair.
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable shipping cost per unit * quantity shipped) plus (fixed activation cost * route activation flag):  
    Minimize sum_{i,j} [c[i,j] * x[i,j] + f[i,j] * y[i,j}].
7.  **Formulate Constraints:**
    -   Market Demand Satisfaction: For each market j, the total shipments received from all depots must equal its demand:  
        sum_{i} x[i,j] = D[j]  for all j.
    -   Depot Supply Capacity: For each depot i, the total shipments sent to all markets must not exceed its capacity:  
        sum_{j} x[i,j] <= S[i]  for all i.
    -   Route Activation Linking: For each route (i,j), shipments are only allowed if the route is activated, and cannot exceed M[i,j]:  
        x[i,j] <= M[i,j] * y[i,j]  for all (i,j).
    -   Nonnegativity: x[i,j] >= 0 for all (i,j).
    -   Binary Route Activation: y[i,j] in {0,1} for all (i,j).
[Abstract Model Plan END]