[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if activated (incurring a fixed cost), and the objective is to minimize the sum of variable transportation costs and fixed route activation costs, subject to plant capacities, retailer demands, and route activation logic.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Plants (from plant_capacity.csv) and Retailers (from retailer_demand.csv). Each route is defined by a (Plant, Retailer) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if the route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacity (`SupplyCapacity[i]`): from plant_capacity.csv, column 'SupplyCapacity'.
    -   Retailer demand (`Demand[j]`): from retailer_demand.csv, column 'Demand'.
    -   Big-M for linking (`M[i,j]`): computed as min(SupplyCapacity[i], Demand[j]) for each (i,j) pair.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable cost per carton * quantity shipped) plus (fixed activation cost * route activation indicator):  
    Minimize sum_{i in Plants, j in Retailers} [c[i,j] * x[i,j] + f[i,j] * y[i,j]].
7.  **Formulate Constraints:**
    -   Retailer demand satisfaction: For each retailer j, sum_{i in Plants} x[i,j] = Demand[j].
    -   Plant supply capacity: For each plant i, sum_{j in Retailers} x[i,j] ≤ SupplyCapacity[i].
    -   Route activation linking: For each (i,j), x[i,j] ≤ M[i,j] * y[i,j], ensuring shipments only occur if the route is activated.
    -   Nonnegativity: For all (i,j), x[i,j] ≥ 0.
    -   Binary route activation: For all (i,j), y[i,j] ∈ {0,1}.
[Abstract Model Plan END]