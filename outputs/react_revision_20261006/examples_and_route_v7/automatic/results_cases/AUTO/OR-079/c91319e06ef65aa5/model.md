[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct and how to route shipments from these factories to 8 distribution centers to minimize the total system cost, which includes both fixed facility construction costs and variable shipping costs, while meeting all distribution center demands and respecting facility capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with fixed charges and transportation costs.
3.  **Define Index Sets:** The primary indices are:
    - Facilities: \( i \in \{\text{A1}, \text{A2}, ..., \text{A15}\} \) (from facility_costs.csv)
    - Distribution Centers: \( j \in \{\text{B1}, \text{B2}, ..., \text{B8}\} \) (from demand_requirements.csv and shipping_costs.csv)
4.  **Define Decision Variables:**
    -   \( y_i \) = 1 if facility \( i \) is constructed, 0 otherwise. Type: GRB.BINARY.
    -   \( x_{ij} \) = quantity shipped from facility \( i \) to distribution center \( j \). Type: GRB.CONTINUOUS (assumed nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Facility fixed costs: from 'FixedCost' column in facility_costs.csv, indexed by 'Facility'.
    -   Facility capacities: from 'Capacity' column in facility_costs.csv, indexed by 'Facility'.
    -   Shipping costs: from shipping_costs.csv, columns 'B1'–'B8', indexed by 'Origin' (facility) and destination.
    -   Distribution center demands: from 'Demand' column in demand_requirements.csv, indexed by 'Destination'.
6.  **Formulate Objective:** Minimize the total system cost, which is the sum of:
    -   The fixed construction costs for all selected facilities: \( \sum_{i} \text{FixedCost}_i \cdot y_i \)
    -   The total shipping costs: \( \sum_{i} \sum_{j} \text{ShippingCost}_{ij} \cdot x_{ij} \)
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each distribution center \( j \), the total shipments received from all facilities must meet its demand:
        -   \( \sum_{i} x_{ij} = \text{Demand}_j \) for all \( j \)
    -   Facility Capacity: For each facility \( i \), the total shipments sent from that facility cannot exceed its capacity if it is constructed:
        -   \( \sum_{j} x_{ij} \leq \text{Capacity}_i \cdot y_i \) for all \( i \)
    -   Facility Activation: \( y_i \in \{0,1\} \) for all \( i \)
    -   Nonnegativity: \( x_{ij} \geq 0 \) for all \( i, j \)
[Abstract Model Plan END]