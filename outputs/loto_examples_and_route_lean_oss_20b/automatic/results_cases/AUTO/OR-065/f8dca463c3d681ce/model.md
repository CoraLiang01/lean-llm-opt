[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which distribution warehouses to activate and how to allocate inventory from those warehouses to independent musicians and bands so that each customer’s demand is satisfied while minimizing the total cost, which includes fixed opening costs for warehouses and per‑unit transportation costs.
2.  **Identify Model Type:** This is a Mixed Integer Programming (MIP) problem, specifically an Uncapacitated Facility Location Problem (UFLP) with fixed‑charge and transportation costs.
3.  **Define Index Sets:**  
    - **Warehouses** \(I = \{S1, S2, S3\}\) (from *fixed_cost.csv* and *transportation_costs.csv*).  
    - **Customers** \(J = \{C1, C2, C3\}\) (from *demand.csv* and *transportation_costs.csv*).
4.  **Define Decision Variables:**  
    - `x[i,j]` = quantity of goods shipped from warehouse \(i\) to customer \(j\). Type: **GRB.CONTINUOUS** (non‑negative).  
    - `y[i]` = 1 if warehouse \(i\) is opened/activated, 0 otherwise. Type: **GRB.BINARY**.
5.  **Identify Parameters (from Schema):**  
    - **Fixed costs** \(f_i\) come from the column *fixed_costs* in *fixed_cost.csv*.  
    - **Transportation costs** per unit \(t_{ij}\) come from the columns *C1*, *C2*, *C3* in *transportation_costs.csv*.  
    - **Demand** \(d_j\) comes from the column *demand* in *demand.csv*.  
    - **Big‑M** \(M_i\) is a sufficiently large number (e.g., the sum of all customer demands or the maximum demand) used to link shipment quantities to warehouse activation.
6.  **Formulate Objective:**  
    Minimize total cost  
    \[
    \min \;\; \sum_{i \in I} f_i \, y_i \;+\; \sum_{i \in I}\sum_{j \in J} t_{ij} \, x_{ij}
    \]
7.  **Formulate Constraints:**  
    - **Demand satisfaction** (for each customer \(j\)):  
      \[
      \sum_{i \in I} x_{ij} \;\ge\; d_j
      \]
    - **Linking (activation) constraints** (for each warehouse \(i\) and customer \(j\)):  
      \[
      x_{ij} \;\le\; M_i \, y_i
      \]
    - **Non‑negativity** (for all \(i,j\)):  
      \[
      x_{ij} \;\ge\; 0
      \]
    - **Binary activation** (for all \(i\)):  
      \[
      y_i \;\in\; \{0,1\}
      \]
[Abstract Model Plan END]