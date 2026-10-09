[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and the shipment quantity on each route is nonnegative and allowed only if the route is activated. The plan must satisfy all retailer demands, not exceed plant capacities, and correctly link shipment quantities to route activation.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are Plants (from plant_capacity.csv) and Retailers (from retailer_demand.csv). The set of routes is the Cartesian product of Plants × Retailers.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if the route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per carton (`c[i,j]`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacity (`SupplyCapacity[i]`): from plant_capacity.csv, column 'SupplyCapacity' for each Plant.
    -   Retailer demand (`Demand[j]`): from retailer_demand.csv, column 'Demand' for each Retailer.
    -   Big-M for linking (`M[i,j]`): computed as min(SupplyCapacity[i], Demand[j]) for each route.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all routes of (variable cost per carton × shipment quantity) plus (fixed activation cost × route activation flag):  
    Minimize ∑_{i in Plants} ∑_{j in Retailers} [c[i,j] * x[i,j] + f[i,j] * y[i,j]]
7.  **Formulate Constraints:**
    -   Retailer demand satisfaction: For each retailer j, the sum of shipments received from all plants must equal Demand[j]:  
        ∑_{i in Plants} x[i,j] = Demand[j]  ∀ j
    -   Plant supply capacity: For each plant i, the total shipments sent to all retailers must not exceed SupplyCapacity[i]:  
        ∑_{j in Retailers} x[i,j] ≤ SupplyCapacity[i]  ∀ i
    -   Route activation linking: For each route (i,j), shipment is allowed only if the route is activated:  
        x[i,j] ≤ M[i,j] * y[i,j]  ∀ i,j
    -   Nonnegativity: x[i,j] ≥ 0  ∀ i,j
    -   Binary restrictions: y[i,j] ∈ {0,1}  ∀ i,j
[Abstract Model Plan END]