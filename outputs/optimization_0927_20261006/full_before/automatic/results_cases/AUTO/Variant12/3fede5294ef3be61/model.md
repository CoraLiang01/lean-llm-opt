[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan for distributing cartons from plants to retailers, where each plant-retailer route incurs both a variable cost per carton shipped and a fixed cost if the route is used. Shipments can only occur on activated routes, and all retailer demands must be met without exceeding plant capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (from plant_capacity.csv: P1, P2, P3)
    - Retailers (from retailer_demand.csv: R1, R2, R3, R4, R5, R6)
    - Routes (all possible plant-retailer pairs)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacities (`SupplyCapacity[i]`): from plant_capacity.csv, column 'SupplyCapacity'.
    -   Retailer demands (`Demand[j]`): from retailer_demand.csv, column 'Demand'.
    -   Linking constraint upper bound (`M[i,j]`): computed as min(SupplyCapacity[i], Demand[j]) for each route.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable cost per carton * number of cartons shipped) plus (fixed cost if the route is used):
    - Minimize: sum over all (i,j) of [c[i,j] * x[i,j] + f[i,j] * y[i,j]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Retailer Demand Satisfaction): For each retailer j, the total cartons received from all plants must equal its demand.
        - sum over i of x[i,j] = Demand[j], for all retailers j.
    -   Constraint 2 (Plant Supply Capacity): For each plant i, the total cartons shipped to all retailers cannot exceed its supply capacity.
        - sum over j of x[i,j] ≤ SupplyCapacity[i], for all plants i.
    -   Constraint 3 (Route Activation Linking): Shipments on a route can only occur if the route is activated; use a big-M constraint.
        - x[i,j] ≤ M[i,j] * y[i,j], for all routes (i,j), where M[i,j] = min(SupplyCapacity[i], Demand[j]).
    -   Constraint 4 (Nonnegativity): x[i,j] ≥ 0 for all (i,j).
    -   Constraint 5 (Binary Route Activation): y[i,j] ∈ {0,1} for all (i,j).
[Abstract Model Plan END]