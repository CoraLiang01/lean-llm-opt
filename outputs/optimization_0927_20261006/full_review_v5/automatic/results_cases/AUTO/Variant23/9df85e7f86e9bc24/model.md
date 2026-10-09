[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipping plan from depots to markets, where each route incurs a fixed activation cost if used and a variable cost per unit shipped. The plan must satisfy all market demands, not exceed depot capacities, and only allow shipments on activated routes, with appropriate linking constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Depots (from depot_capacity.csv) and Markets (from market_demand.csv). Each route is defined by a (Depot, Market) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity shipped from depot i to market j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from depot i to market j is activated, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping cost per unit (`c[i,j]`): from route_variable_costs.csv, columns 'M1'–'M5' for each depot.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns 'M1'–'M5' for each depot.
    -   Depot supply capacity (`S[i]`): from depot_capacity.csv, column 'SupplyCapacity'.
    -   Market demand (`D[j]`): from market_demand.csv, column 'Demand'.
    -   Route upper bound (`M[i,j]`): computed as min(S[i], D[j]) for each (i,j) pair.
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable shipping cost per unit * quantity shipped) plus (fixed activation cost * route activation flag):  
    Minimize sum_{i,j} [c[i,j] * x[i,j] + f[i,j] * y[i,j]].
7.  **Formulate Constraints:**
    -   Market Demand Satisfaction: For each market j, sum over all depots i of x[i,j] = D[j] (each market's demand must be fully met).
    -   Depot Supply Limit: For each depot i, sum over all markets j of x[i,j] ≤ S[i] (do not exceed depot capacity).
    -   Route Activation Linking: For each route (i,j), x[i,j] ≤ M[i,j] * y[i,j] (no shipment unless route is activated; shipment cannot exceed min(depot capacity, market demand)).
    -   Nonnegativity: For all (i,j), x[i,j] ≥ 0.
    -   Binary Route Activation: For all (i,j), y[i,j] ∈ {0,1}.
[Abstract Model Plan END]