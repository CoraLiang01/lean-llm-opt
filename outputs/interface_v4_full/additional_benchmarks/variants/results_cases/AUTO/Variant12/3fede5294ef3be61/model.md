[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and the shipment quantity on each route is subject to plant capacities, retailer demands, and route activation. The model must include both variable and fixed costs, and enforce supply, demand, and linking constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (from plant_capacity.csv): P = {P1, P2, P3}
    - Retailers (from retailer_demand.csv): R = {R1, R2, R3, R4, R5, R6}
    - Routes: All combinations of (Plant, Retailer) pairs (i, j) where i ∈ P, j ∈ R
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacity (`SupplyCapacity[i]`): from plant_capacity.csv, column 'SupplyCapacity' for each Plant.
    -   Retailer demand (`Demand[j]`): from retailer_demand.csv, column 'Demand' for each Retailer.
    -   Linking constraint upper bound (`M[i,j]`): computed as min(SupplyCapacity[i], Demand[j]) for each route (i, j).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable cost per carton * shipment quantity) plus (fixed cost * route activation binary):
        Minimize sum_{i in P, j in R} [ c[i,j] * x[i,j] + f[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Retailer Demand Satisfaction): For each retailer j, the total cartons received from all plants must equal its demand:
            sum_{i in P} x[i,j] = Demand[j]   for all j in R
    -   Constraint 2 (Plant Supply Capacity): For each plant i, the total cartons shipped to all retailers cannot exceed its capacity:
            sum_{j in R} x[i,j] <= SupplyCapacity[i]   for all i in P
    -   Constraint 3 (Route Activation Linking): For each route (i, j), shipments can only occur if the route is activated, and cannot exceed the minimum of plant capacity and retailer demand:
            x[i,j] <= M[i,j] * y[i,j]   for all i in P, j in R
    -   Constraint 4 (Nonnegativity): x[i,j] >= 0   for all i in P, j in R
    -   Constraint 5 (Binary Route Activation): y[i,j] ∈ {0,1}   for all i in P, j in R
[Abstract Model Plan END]