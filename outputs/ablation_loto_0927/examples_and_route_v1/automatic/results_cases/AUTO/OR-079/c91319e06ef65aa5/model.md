[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct and how to route shipments from these factories to 8 distribution centers to minimize the total system cost, which includes both fixed facility investment and variable shipping costs, while meeting all distribution center demands and respecting facility capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem (fixed-charge network design).
3.  **Define Index Sets:** The primary indices are:
    - Facilities: \( i \in \{\text{A1}, \ldots, \text{A15}\} \)
    - Distribution Centers: \( j \in \{\text{B1}, \ldots, \text{B8}\} \)
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if facility \( i \) is constructed, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = quantity shipped from facility \( i \) to distribution center \( j \). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Facility fixed costs: from facility_costs.csv, column 'FixedCost' (indexed by 'Facility').
    -   Facility capacities: from facility_costs.csv, column 'Capacity' (indexed by 'Facility').
    -   Shipping costs: from shipping_costs.csv, columns 'B1'–'B8' (indexed by 'Origin' and destination).
    -   Distribution center demands: from demand_requirements.csv, column 'Demand' (indexed by 'Destination').
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    -   The fixed costs for each constructed facility: sum over \( i \) of FixedCost[i] * y[i]
    -   The total shipping costs: sum over \( i, j \) of ShippingCost[i,j] * x[i,j]
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each distribution center \( j \), the sum of shipments received from all facilities must equal its demand (from demand_requirements.csv):  
        sum over \( i \) of x[i,j] = Demand[j]
    -   Facility Capacity: For each facility \( i \), the total amount shipped from that facility cannot exceed its capacity if it is constructed:  
        sum over \( j \) of x[i,j] ≤ Capacity[i] * y[i]
    -   Linking: Shipments from a facility are only allowed if the facility is constructed (enforced by the above capacity constraint).
    -   Nonnegativity: All shipment variables x[i,j] ≥ 0.
    -   Binary: All facility decision variables y[i] ∈ {0,1}.
[Abstract Model Plan END]