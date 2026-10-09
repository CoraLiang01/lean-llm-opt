[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipping plan for distributing cartons from plants to retailers, where each plant-retailer route has both a variable per-carton cost and a fixed activation cost. Shipments can only occur on a route if it is activated. The plan must satisfy all retailer demands, not exceed plant capacities, and properly link route activation to shipment quantities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (from plant_capacity.csv): P1, P2, P3
    - Retailers (from retailer_demand.csv): R1, R2, R3, R4, R5, R6
    - Routes: All possible (Plant, Retailer) pairs (i, j)
4.  **Define Decision Variables:**
    -   `x_ij` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y_ij` = 1 if route from plant i to retailer j is activated (opened), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable cost per carton for each route (i, j): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed cost for activating each route (i, j): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacities: from plant_capacity.csv, column 'SupplyCapacity'.
    -   Retailer demands: from retailer_demand.csv, column 'Demand'.
    -   Linking constraint upper bound M_ij: for each route (i, j), set as min(plant i's capacity, retailer j's demand).
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable cost per carton * number of cartons shipped) plus (fixed cost * route activation indicator):
    - Minimize: sum over all (i, j) of [variable_cost_ij * x_ij + fixed_cost_ij * y_ij]
7.  **Formulate Constraints:**
    -   Constraint 1 (Retailer Demand Satisfaction): For each retailer j, the total cartons received from all plants must equal its demand.
        - For all j: sum over i of x_ij = demand_j
    -   Constraint 2 (Plant Capacity Limit): For each plant i, the total cartons shipped to all retailers cannot exceed its supply capacity.
        - For all i: sum over j of x_ij ≤ capacity_i
    -   Constraint 3 (Route Activation Linking): For each route (i, j), shipments can only occur if the route is activated, and cannot exceed M_ij.
        - For all (i, j): x_ij ≤ M_ij * y_ij, where M_ij = min(capacity_i, demand_j)
    -   Constraint 4 (Nonnegativity): For all (i, j), x_ij ≥ 0
    -   Constraint 5 (Binary Route Activation): For all (i, j), y_ij ∈ {0, 1}
[Abstract Model Plan END]