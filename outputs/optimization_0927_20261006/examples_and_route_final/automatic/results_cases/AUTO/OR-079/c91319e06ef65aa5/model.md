[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct and how to route shipments from these factories to 8 distribution centers to minimize the total system cost, which includes both fixed facility construction costs and variable shipping costs, while meeting all distribution center demands and respecting facility capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with fixed-charge and transportation components.
3.  **Define Index Sets:** The primary indices are:
    - Facilities: \( i \in \text{Facilities} \) (A1–A15, from facility_costs.csv)
    - Distribution Centers: \( j \in \text{Distribution Centers} \) (B1–B8, from demand_requirements.csv)
4.  **Define Decision Variables:**
    -   \( y_i \) = 1 if facility \( i \) is constructed, 0 otherwise. Type: GRB.BINARY.
    -   \( x_{ij} \) = quantity shipped from facility \( i \) to distribution center \( j \). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Facility fixed costs: 'FixedCost' column in facility_costs.csv.
    -   Facility capacities: 'Capacity' column in facility_costs.csv.
    -   Shipping costs: 'B1'–'B8' columns in shipping_costs.csv, indexed by 'Origin'.
    -   Distribution center demands: 'Demand' column in demand_requirements.csv, indexed by 'Destination'.
6.  **Formulate Objective:** Minimize the total system cost, which is the sum of:
    -   The fixed construction costs for all selected facilities (\( \sum_i \text{FixedCost}_i \cdot y_i \))
    -   The total shipping costs for all shipments (\( \sum_{i,j} \text{ShippingCost}_{ij} \cdot x_{ij} \))
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each distribution center \( j \), the total shipments received from all facilities must meet its demand (\( \sum_i x_{ij} = \text{Demand}_j \)).
    -   Facility Capacity: For each facility \( i \), the total shipments sent from that facility cannot exceed its capacity if constructed (\( \sum_j x_{ij} \leq \text{Capacity}_i \cdot y_i \)).
    -   Linking: Shipments from a facility are only allowed if the facility is constructed (enforced by the capacity constraint above).
    -   Nonnegativity: All shipment variables \( x_{ij} \geq 0 \).
    -   Binary: All facility construction variables \( y_i \in \{0,1\} \).
[Abstract Model Plan END]