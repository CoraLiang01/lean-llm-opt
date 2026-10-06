[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost fixed-charge transportation model for shipping cartons from plants to retailers, where each plant-retailer route has both a variable per-unit shipping cost and a fixed activation cost. Shipments on a route are only allowed if the route is activated. The model must ensure all retailer demands are met, plant capacities are not exceeded, and route activation is properly linked to shipments.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (fixed-cost) transportation structure.
3.  **Define Index Sets:** The primary indices are:
    - Plants (from plant_capacity.csv): P = {P1, P2, P3}
    - Retailers (from retailer_demand.csv): R = {R1, R2, R3, R4, R5, R6}
    - Routes: All combinations of (Plant, Retailer) pairs (i, j) where i ∈ P, j ∈ R
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from plant i to retailer j is activated (opened), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping cost per carton for each route (i,j): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed activation cost for each route (i,j): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacities: from plant_capacity.csv, column 'SupplyCapacity' for each Plant.
    -   Retailer demands: from retailer_demand.csv, column 'Demand' for each Retailer.
    -   Linking constraint upper bound M_ij: for each route (i,j), M_ij = min(SupplyCapacity of plant i, Demand of retailer j).
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable shipping cost per carton * number of cartons shipped) plus (fixed activation cost * route activation binary):
        Minimize sum_{i in P, j in R} [ variable_cost[i,j] * x[i,j] + fixed_cost[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Retailer Demand Satisfaction): For each retailer j, the total cartons received from all plants must equal its demand:
            sum_{i in P} x[i,j] = Demand[j]   for all j in R
    -   Constraint 2 (Plant Supply Capacity): For each plant i, the total cartons shipped to all retailers cannot exceed its supply capacity:
            sum_{j in R} x[i,j] <= SupplyCapacity[i]   for all i in P
    -   Constraint 3 (Route Activation Linking): For each route (i,j), shipments are only allowed if the route is activated:
            x[i,j] <= M_ij * y[i,j]   for all i in P, j in R
        where M_ij = min(SupplyCapacity[i], Demand[j])
    -   Constraint 4 (Nonnegativity): x[i,j] >= 0   for all i in P, j in R
    -   Constraint 5 (Binary Route Activation): y[i,j] ∈ {0,1}   for all i in P, j in R
[Abstract Model Plan END]