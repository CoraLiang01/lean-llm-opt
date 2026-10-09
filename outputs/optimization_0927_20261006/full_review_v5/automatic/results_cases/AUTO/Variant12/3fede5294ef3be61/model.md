[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and the shipment quantity on each route is nonnegative and linked to route activation. The plan must satisfy all retailer demands, not exceed plant capacities, and minimize the sum of variable and fixed route costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Plants (from 'Plant' in plant_capacity.csv) and Retailers (from 'Retailer' in retailer_demand.csv). Routes are defined for each (Plant, Retailer) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from 'route_variable_costs.csv', columns 'R1'–'R6' for each plant.
    -   Fixed route activation cost (`f[i,j]`): from 'route_fixed_costs.csv', columns 'R1'–'R6' for each plant.
    -   Plant supply capacity (`S[i]`): from 'SupplyCapacity' in plant_capacity.csv.
    -   Retailer demand (`D[j]`): from 'Demand' in retailer_demand.csv.
    -   Maximum possible shipment on route (`M[i,j]`): computed as min(S[i], D[j]) for each (i,j) pair.
6.  **Formulate Objective:** Minimize total cost, i.e., sum over all routes of (variable cost per carton * shipment quantity) plus (fixed activation cost * route activation flag):  
    Minimize ∑_{i,j} [c[i,j] * x[i,j] + f[i,j] * y[i,j}].
7.  **Formulate Constraints:**
    -   Retailer demand satisfaction: For each retailer j, sum over all plants i of x[i,j] = D[j].
    -   Plant supply capacity: For each plant i, sum over all retailers j of x[i,j] ≤ S[i].
    -   Route activation linking: For each route (i,j), x[i,j] ≤ M[i,j] * y[i,j].
    -   Nonnegativity: For all (i,j), x[i,j] ≥ 0.
    -   Binary route activation: For all (i,j), y[i,j] ∈ {0,1}.
[Abstract Model Plan END]