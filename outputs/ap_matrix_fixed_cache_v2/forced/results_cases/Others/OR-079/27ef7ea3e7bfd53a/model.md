[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct (incurring fixed costs and respecting capacities), and how to route shipments from these selected factories to 8 distribution centers to meet fixed demand at minimum total system cost (sum of facility fixed costs and shipping costs).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with fixed-charge and transportation components.
3.  **Define Index Sets:** The primary indices are:
    - Facilities (i ∈ Facilities, from 'Facility' in facility_costs.csv)
    - Distribution Centers (j ∈ DistributionCenters, from 'Destination' in demand_requirements.csv and columns B1–B8 in shipping_costs.csv)
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if facility i is constructed, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = quantity shipped from facility i to distribution center j. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Facility fixed costs: 'FixedCost' column in facility_costs.csv.
    -   Facility capacities: 'Capacity' column in facility_costs.csv.
    -   Shipping costs per unit: columns B1–B8 in shipping_costs.csv, indexed by facility (Origin) and distribution center.
    -   Demand at each distribution center: 'Demand' column in demand_requirements.csv.
6.  **Formulate Objective:** Minimize the total system cost, which is the sum of:
    -   The fixed costs for each constructed facility (sum over i of FixedCost[i] * y[i])
    -   The total shipping costs (sum over i and j of ShippingCost[i,j] * x[i,j])
7.  **Formulate Constraints:**
    -   Demand satisfaction: For each distribution center j, the sum over all facilities i of x[i,j] must equal Demand[j].
    -   Facility capacity: For each facility i, the sum over all distribution centers j of x[i,j] must be less than or equal to Capacity[i] * y[i] (i.e., only constructed facilities can ship, and cannot exceed their capacity).
    -   Nonnegativity: All shipment variables x[i,j] ≥ 0.
    -   Binary: All facility construction variables y[i] ∈ {0,1}.
[Abstract Model Plan END]