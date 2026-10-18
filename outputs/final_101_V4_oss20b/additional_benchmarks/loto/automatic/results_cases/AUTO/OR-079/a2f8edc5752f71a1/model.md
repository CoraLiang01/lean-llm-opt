[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to minimize the total system cost by deciding which of 15 potential factory sites to build and how to ship goods from the selected factories to 8 distribution centers so that all fixed demand is satisfied.
2.  **Identify Model Type:** This is a Capacitated Facility Location Problem (CFLP) with fixed facility costs and variable shipping costs.
3.  **Define Index Sets:**  
   - **Facilities** \(I = \{A1, A2, \dots, A15\}\)  
   - **Destinations** \(J = \{B1, B2, \dots, B8\}\)
4.  **Define Decision Variables:**  
    - `y[i]` = 1 if facility *i* is built, 0 otherwise. Type: `GRB.BINARY`.  
    - `x[i,j]` = quantity shipped from facility *i* to destination *j*. Type: `GRB.CONTINUOUS` (≥ 0).
5.  **Identify Parameters (from Schema):**  
    - `FixedCost[i]` from `facility_costs.csv` column **FixedCost**.  
    - `Capacity[i]` from `facility_costs.csv` column **Capacity**.  
    - `ShippingCost[i,j]` from `shipping_costs.csv` columns **B1**–**B8**.  
    - `Demand[j]` from `demand_requirements.csv` column **Demand**.
6.  **Formulate Objective:**  
    Minimize  
    \[
    \sum_{i \in I} \text{FixedCost}[i] \cdot y[i] \;+\; \sum_{i \in I}\sum_{j \in J} \text{ShippingCost}[i,j] \cdot x[i,j]
    \]
7.  **Formulate Constraints:**  
    - **Demand satisfaction** (for each destination):  
      \[
      \sum_{i \in I} x[i,j] \;=\; \text{Demand}[j] \quad \forall j \in J
      \]
    - **Facility capacity** (for each facility):  
      \[
      \sum_{j \in J} x[i,j] \;\le\; \text{Capacity}[i] \cdot y[i] \quad \forall i \in I
      \]
    - **Non‑negativity**:  
      \[
      x[i,j] \;\ge\; 0 \quad \forall i \in I,\; j \in J
      \]
    - **Binary**:  
      \[
      y[i] \in \{0,1\} \quad \forall i \in I
      \]
[Abstract Model Plan END]