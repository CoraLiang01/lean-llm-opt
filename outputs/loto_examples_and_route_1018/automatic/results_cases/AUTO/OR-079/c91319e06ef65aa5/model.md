[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct and how to route shipments from these factories to 8 distribution centers to minimize the total system cost, which includes both fixed facility construction costs and variable shipping costs, while meeting all distribution center demands and respecting facility capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with fixed-charge and transportation components.
3.  **Define Index Sets:** The primary indices are:
    - Facilities (i ∈ Facilities, from 'Facility' in facility_costs.csv)
    - Distribution Centers (j ∈ DistributionCenters, from 'Destination' in demand_requirements.csv and columns B1–B8 in shipping_costs.csv)
4.  **Define Decision Variables:**
    - `y[i]` = 1 if facility i is constructed, 0 otherwise. Type: GRB.BINARY.
    - `x[i,j]` = quantity shipped from facility i to distribution center j. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Facility fixed costs: 'FixedCost' from facility_costs.csv.
    - Facility capacities: 'Capacity' from facility_costs.csv.
    - Shipping costs per unit: entries in shipping_costs.csv (columns B1–B8 for each 'Origin').
    - Distribution center demands: 'Demand' from demand_requirements.csv.
6.  **Formulate Objective:** Minimize the total system cost, which is the sum of fixed construction costs for selected facilities plus the total shipping costs for all shipments from open facilities to distribution centers. That is, minimize sum over i of (FixedCost[i] * y[i]) plus sum over i,j of (ShippingCost[i,j] * x[i,j]).
7.  **Formulate Constraints:**
    - Demand Satisfaction: For each distribution center j, the sum over all facilities i of x[i,j] must equal Demand[j] (i.e., all demand at each distribution center must be met).
    - Facility Capacity: For each facility i, the sum over all distribution centers j of x[i,j] must be less than or equal to Capacity[i] * y[i] (i.e., a facility can only ship if constructed, and cannot exceed its capacity).
    - Nonnegativity: All shipment variables x[i,j] ≥ 0.
    - Binary Facility Selection: All y[i] ∈ {0,1}.
[Abstract Model Plan END]