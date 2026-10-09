[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan for distributing cartons from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and each shipment incurs a variable per-carton cost. The plan must satisfy all retailer demands, not exceed plant capacities, and ensure that shipments on a route are only possible if the route is activated.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (from plant_capacity.csv): P1, P2, P3
    - Retailers (from retailer_demand.csv): R1, R2, R3, R4, R5, R6
    - Routes: All combinations of (Plant, Retailer)
4.  **Define Decision Variables:**
    -   `x_ij` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y_ij` = 1 if route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c_ij`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f_ij`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacity (`SupplyCapacity_i`): from plant_capacity.csv, column 'SupplyCapacity' for each Plant.
    -   Retailer demand (`Demand_j`): from retailer_demand.csv, column 'Demand' for each Retailer.
    -   Linking constraint upper bound (`M_ij`): computed as min(SupplyCapacity_i, Demand_j) for each route (i,j).
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable cost per carton * quantity shipped) plus (fixed activation cost * route activation binary):
    - Minimize: sum over all (i,j) of [c_ij * x_ij + f_ij * y_ij]
7.  **Formulate Constraints:**
    -   Constraint 1 (Retailer Demand Satisfaction): For each retailer j, the total cartons received from all plants must equal its demand.
        - sum over i of x_ij = Demand_j, for all j
    -   Constraint 2 (Plant Supply Capacity): For each plant i, the total cartons shipped to all retailers cannot exceed its supply capacity.
        - sum over j of x_ij ≤ SupplyCapacity_i, for all i
    -   Constraint 3 (Route Activation Linking): For each route (i,j), shipments can only occur if the route is activated; i.e., x_ij ≤ M_ij * y_ij, where M_ij = min(SupplyCapacity_i, Demand_j).
    -   Constraint 4 (Nonnegativity): x_ij ≥ 0 for all (i,j).
    -   Constraint 5 (Binary Route Activation): y_ij ∈ {0,1} for all (i,j).
[Abstract Model Plan END]