[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and the shipment quantity on each route is nonnegative and allowed only if the route is activated. The plan must satisfy all retailer demands, not exceed plant capacities, and correctly link shipment quantities to route activation.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Plants (from plant_capacity.csv) and Retailers (from retailer_demand.csv), forming the set of all possible plant-retailer routes.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity of cartons shipped from plant `i` to retailer `j`. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from plant `i` to retailer `j` is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from route_variable_costs.csv, columns 'R1'–'R6' for each plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns 'R1'–'R6' for each plant.
    -   Plant supply capacity (`S[i]`): from plant_capacity.csv, column 'SupplyCapacity'.
    -   Retailer demand (`D[j]`): from retailer_demand.csv, column 'Demand'.
    -   Big-M for linking (`M[i,j]`): set as min(S[i], D[j]) for each route.
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable cost per carton * shipment quantity) plus (fixed activation cost * route activation flag):  
        Minimize sum_{i,j} [ c[i,j] * x[i,j] + f[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    -   Retailer demand satisfaction: For each retailer `j`, sum over all plants `i` of x[i,j] == D[j] (each retailer receives exactly its demand).
    -   Plant supply capacity: For each plant `i`, sum over all retailers `j` of x[i,j] <= S[i] (each plant does not exceed its capacity).
    -   Route activation linking: For each route (i,j), x[i,j] <= M[i,j] * y[i,j] (shipment only allowed if route is activated; upper-bounded by min(plant capacity, retailer demand)).
    -   Nonnegativity: For all (i,j), x[i,j] >= 0.
    -   Binary restrictions: For all (i,j), y[i,j] in {0,1}.
[Abstract Model Plan END]