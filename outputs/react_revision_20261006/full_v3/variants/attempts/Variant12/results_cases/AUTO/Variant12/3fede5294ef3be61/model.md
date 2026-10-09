[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and the shipment on each route is limited by plant capacity and retailer demand. The model must ensure all retailer demands are met, no plant exceeds its capacity, and route activation is properly linked to shipment quantities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (from plant_capacity.csv): P1, P2, P3
    - Retailers (from retailer_demand.csv): R1, R2, R3, R4, R5, R6
    - Routes: All possible (Plant, Retailer) pairs (i, j)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacity (`SupplyCapacity[i]`): from plant_capacity.csv, column 'SupplyCapacity'.
    -   Retailer demand (`Demand[j]`): from retailer_demand.csv, column 'Demand'.
    -   Big-M for linking (`M[i,j]`): computed as min(SupplyCapacity[i], Demand[j]) for each route (i,j).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable cost per carton * shipment quantity) plus (fixed cost * route activation indicator):
    - Minimize: sum over all (i,j) of [c[i,j] * x[i,j] + f[i,j] * y[i,j]]
7.  **Formulate Constraints:**
    -   **Retailer Demand Satisfaction:** For each retailer j, the total cartons received from all plants must equal its demand:
        - sum over i of x[i,j] = Demand[j]   for all j
    -   **Plant Supply Capacity:** For each plant i, the total cartons shipped to all retailers cannot exceed its capacity:
        - sum over j of x[i,j] ≤ SupplyCapacity[i]   for all i
    -   **Route Activation Linking:** For each route (i,j), shipments can only occur if the route is activated, and cannot exceed the minimum of plant capacity and retailer demand:
        - x[i,j] ≤ M[i,j] * y[i,j]   for all (i,j)
    -   **Nonnegativity:** x[i,j] ≥ 0   for all (i,j)
    -   **Binary Route Activation:** y[i,j] ∈ {0,1}   for all (i,j)
[Abstract Model Plan END]