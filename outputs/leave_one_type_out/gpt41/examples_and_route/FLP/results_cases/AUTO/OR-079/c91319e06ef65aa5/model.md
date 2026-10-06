[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct (incurring fixed costs and respecting capacities), and how to route shipments from these selected factories to 8 distribution centers to meet fixed demands at minimum total cost (fixed facility costs plus variable shipping costs).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem (fixed-charge network design).
3.  **Define Index Sets:** The primary indices are:
    - Facilities: \( i \in \{\text{A1}, \ldots, \text{A15}\} \) (from facility_costs.csv)
    - Distribution Centers: \( j \in \{\text{B1}, \ldots, \text{B8}\} \) (from demand_requirements.csv)
4.  **Define Decision Variables:**
    -   \( y_i \) = 1 if facility \( i \) is constructed, 0 otherwise. Type: GRB.BINARY.
    -   \( x_{ij} \) = quantity shipped from facility \( i \) to distribution center \( j \). Type: GRB.CONTINUOUS (assumed nonnegative; integer if required by context).
5.  **Identify Parameters (from Schema):**
    -   Facility fixed costs: from 'FixedCost' column in facility_costs.csv.
    -   Facility capacities: from 'Capacity' column in facility_costs.csv.
    -   Shipping costs: from shipping_costs.csv, columns 'B1'–'B8' for each 'Origin' (facility).
    -   Distribution center demands: from 'Demand' column in demand_requirements.csv.
6.  **Formulate Objective:** Minimize total system cost, which is the sum of:
    -   Total fixed costs for constructed facilities: \( \sum_{i} \text{FixedCost}_i \cdot y_i \)
    -   Total shipping costs: \( \sum_{i} \sum_{j} \text{ShippingCost}_{ij} \cdot x_{ij} \)
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each distribution center \( j \), the total shipments received from all facilities must meet its demand:
        -   \( \sum_{i} x_{ij} = \text{Demand}_j \) for all \( j \)
    -   Facility Capacity: For each facility \( i \), the total shipments sent cannot exceed its capacity if constructed:
        -   \( \sum_{j} x_{ij} \leq \text{Capacity}_i \cdot y_i \) for all \( i \)
    -   Linking: Shipments from a facility are only allowed if the facility is constructed (enforced by the above capacity constraint).
    -   Nonnegativity: \( x_{ij} \geq 0 \) for all \( i, j \)
    -   Binary: \( y_i \in \{0,1\} \) for all \( i \)
[Abstract Model Plan END]