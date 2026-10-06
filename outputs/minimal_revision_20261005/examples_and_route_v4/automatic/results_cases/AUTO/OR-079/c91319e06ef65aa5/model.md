[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which subset of 15 potential factory sites to construct and how to route shipments from these factories to 8 distribution centers to minimize the total system cost, which includes both fixed facility investment and variable shipping costs, while meeting all distribution center demands and respecting facility capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location problem with fixed-charge and transportation components.
3.  **Define Index Sets:** The primary indices are:
    - Facilities: \( i \in \{\text{A1}, \text{A2}, ..., \text{A15}\} \) (from facility_costs.csv)
    - Distribution Centers: \( j \in \{\text{B1}, \text{B2}, ..., \text{B8}\} \) (from demand_requirements.csv)
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if facility \( i \) is constructed, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = quantity shipped from facility \( i \) to distribution center \( j \). Type: GRB.CONTINUOUS (assume nonnegative and divisible shipments).
5.  **Identify Parameters (from Schema):**
    -   Facility fixed costs: from 'FixedCost' column in facility_costs.csv.
    -   Facility capacities: from 'Capacity' column in facility_costs.csv.
    -   Shipping costs: from shipping_costs.csv, columns 'B1' to 'B8' for each 'Origin' (facility).
    -   Distribution center demands: from 'Demand' column in demand_requirements.csv.
6.  **Formulate Objective:** Minimize the total system cost, which is the sum of:
    - The fixed costs for all constructed facilities: \( \sum_{i} \text{FixedCost}[i] \cdot y[i] \)
    - The total shipping costs: \( \sum_{i} \sum_{j} \text{ShippingCost}[i,j] \cdot x[i,j] \)
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each distribution center \( j \), the total shipments received from all facilities must meet its demand:
        - \( \sum_{i} x[i,j] = \text{Demand}[j] \) for all \( j \)
    -   Facility Capacity: For each facility \( i \), the total shipments sent cannot exceed its capacity if it is constructed:
        - \( \sum_{j} x[i,j] \leq \text{Capacity}[i] \cdot y[i] \) for all \( i \)
    -   Linking: Shipments from a facility are only allowed if the facility is constructed (enforced by the above capacity constraint).
    -   Nonnegativity: \( x[i,j] \geq 0 \) for all \( i, j \)
    -   Binary: \( y[i] \in \{0,1\} \) for all \( i \)
[Abstract Model Plan END]