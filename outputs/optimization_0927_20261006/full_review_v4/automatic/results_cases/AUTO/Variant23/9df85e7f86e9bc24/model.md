[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipping plan from depots to markets, where each route incurs a fixed activation cost if used and a variable cost per unit shipped. The plan must satisfy all market demands, not exceed depot capacities, and only allow shipments on activated routes, with appropriate linking constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a fixed-charge transportation model.
3.  **Define Index Sets:** The primary indices are:
    - Depots (from depot_capacity.csv, column 'Depot')
    - Markets (from market_demand.csv, column 'Market')
    - Routes (all depot-market pairs, i.e., Cartesian product of Depots × Markets)
4.  **Define Decision Variables:**
    - `x[i,j]` = quantity shipped from depot i to market j. Type: GRB.CONTINUOUS, x[i,j] ≥ 0.
    - `y[i,j]` = 1 if route from depot i to market j is activated, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Variable shipping cost per unit (`c[i,j]`): from route_variable_costs.csv, columns 'M1'–'M5' for each depot.
    - Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns 'M1'–'M5' for each depot.
    - Depot supply capacity (`S[i]`): from depot_capacity.csv, column 'SupplyCapacity' for each depot.
    - Market demand (`D[j]`): from market_demand.csv, column 'Demand' for each market.
    - Linking constraint upper bound (`M[i,j]`): for each route, M[i,j] = min(S[i], D[j]).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable shipping cost per unit × quantity shipped) plus (fixed activation cost × route activation flag):
       Minimize sum_{i in Depots, j in Markets} [ c[i,j] * x[i,j] + f[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    - Constraint 1 (Market Demand Satisfaction): For each market j, the total shipments received from all depots must equal its demand:
          sum_{i in Depots} x[i,j] = D[j]   for all j in Markets
    - Constraint 2 (Depot Supply Capacity): For each depot i, the total shipments sent to all markets must not exceed its capacity:
          sum_{j in Markets} x[i,j] ≤ S[i]   for all i in Depots
    - Constraint 3 (Route Activation Linking): For each route (i,j), shipments are only allowed if the route is activated, and cannot exceed M[i,j]:
          x[i,j] ≤ M[i,j] * y[i,j]   for all i in Depots, j in Markets
    - Constraint 4 (Nonnegativity): x[i,j] ≥ 0   for all i in Depots, j in Markets
    - Constraint 5 (Binary Route Activation): y[i,j] ∈ {0,1}   for all i in Depots, j in Markets
[Abstract Model Plan END]