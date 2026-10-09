[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum-cost shipment plan from plants to retailers, where each plant-retailer route can be used only if a fixed activation cost is paid, and variable per-carton shipping costs apply. Shipments must meet all retailer demands without exceeding plant capacities, and route activation is linked to shipment quantities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) fixed-charge transportation problem.
3.  **Define Index Sets:** The primary indices are Plants (from plant_capacity.csv) and Retailers (from retailer_demand.csv). Each route is defined by a (Plant, Retailer) pair.
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of cartons shipped from plant i to retailer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if the route from plant i to retailer j is activated (used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable shipping cost per carton (`c[i,j]`): from route_variable_costs.csv, columns R1–R6 for each Plant.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns R1–R6 for each Plant.
    -   Plant supply capacity (`SupplyCapacity[i]`): from plant_capacity.csv, column 'SupplyCapacity'.
    -   Retailer demand (`Demand[j]`): from retailer_demand.csv, column 'Demand'.
    -   Big-M for linking (`M[i,j]`): computed as min(SupplyCapacity[i], Demand[j]) for each (i,j) pair.
6.  **Formulate Objective:** Minimize total cost, which is the sum over all routes of (variable cost per carton × shipment quantity) plus (fixed activation cost × route activation variable):  
    Minimize ∑_{i,j} [c[i,j] * x[i,j] + f[i,j] * y[i,j]].
7.  **Formulate Constraints:**
    -   Retailer Demand Satisfaction: For each retailer j, the sum of shipments received from all plants must equal Demand[j]:  
        ∑_{i} x[i,j] = Demand[j].
    -   Plant Supply Capacity: For each plant i, the total shipments sent to all retailers cannot exceed SupplyCapacity[i]:  
        ∑_{j} x[i,j] ≤ SupplyCapacity[i].
    -   Route Activation Linking: For each route (i,j), shipments can only occur if the route is activated:  
        x[i,j] ≤ M[i,j] * y[i,j].
    -   Nonnegativity: For all (i,j), x[i,j] ≥ 0.
    -   Binary Route Activation: For all (i,j), y[i,j] ∈ {0,1}.
[Abstract Model Plan END]