[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipping plan from depots to markets, where each route incurs a fixed activation cost if used and a variable cost per unit shipped. The plan must satisfy all market demands, not exceed depot capacities, and only allow shipments on activated routes.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Depots (from depot_capacity.csv) and Markets (from market_demand.csv). Each route is defined by a (Depot, Market) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity shipped from depot i to market j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from depot i to market j is activated, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping costs per unit: from route_variable_costs.csv, columns 'M1'–'M5' for each depot.
    -   Fixed route activation costs: from route_fixed_costs.csv, columns 'M1'–'M5' for each depot.
    -   Depot supply capacities: from depot_capacity.csv, column 'SupplyCapacity'.
    -   Market demands: from market_demand.csv, column 'Demand'.
    -   Linking constraint upper bound M_ij: computed as min(SupplyCapacity of depot i, Demand of market j).
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable shipping cost per unit * shipment quantity) plus (fixed activation cost * route activation binary).
7.  **Formulate Constraints:**
    -   Market Demand Satisfaction: For each market j, the sum of shipments received from all depots equals the demand of market j.
    -   Depot Supply Limit: For each depot i, the sum of shipments sent to all markets does not exceed the supply capacity of depot i.
    -   Route Activation Linking: For each route (i,j), the shipment quantity x[i,j] is less than or equal to M_ij times y[i,j], ensuring shipments only occur if the route is activated and do not exceed feasible maximums.
    -   Nonnegativity: All shipment variables x[i,j] are greater than or equal to zero.
    -   Binary Restrictions: All route activation variables y[i,j] are binary (0 or 1).
[Abstract Model Plan END]