[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which liquor suppliers to activate (open) so that the demand of every licensed store is satisfied while minimizing the total cost, which consists of supplier fixed opening costs and per‑unit transportation costs.

2.  **Identify Model Type:** This is a **Mixed Integer Programming (MIP)** model of the **Uncapacitated Facility Location Problem (UFLP)** with fixed‑charge (opening) costs and linear transportation costs.

3.  **Define Index Sets:**  
    - **Suppliers** \(I = \{ \text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES} \}\) (rows of *fixed_cost.csv* and *transportation_costs.csv*).  
    - **Stores** \(J = \{ \text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT} \}\) (columns of *transportation_costs.csv* and rows of *demand.csv*).

4.  **Define Decision Variables:**  
    - `x[i,j]` = quantity of liquor shipped from supplier \(i \in I\) to store \(j \in J\).  
      Type: `GRB.CONTINUOUS`, \(x[i,j] \ge 0\).  
    - `y[i]` = 1 if supplier \(i\) is opened, 0 otherwise.  
      Type: `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
    - **Fixed opening costs**: `fixed_cost[i]` from *fixed_cost.csv* (column *fixed_costs*).  
    - **Transportation costs per unit**: `trans_cost[i,j]` from *transportation_costs.csv* (row *Unnamed: 0* = supplier, column = store).  
    - **Store demand**: `demand[j]` from *demand.csv* (column *demand*).  
    - **Big‑M value**: `M` chosen as the total demand across all stores, \(M = \sum_{j \in J} demand[j]\), to link shipment variables to opening decisions.

6.  **Formulate Objective:**  
   Minimize total cost  
   \[
   \min \; \sum_{i \in I} \text{fixed_cost}[i] \, y[i]
          \;+\; \sum_{i \in I}\sum_{j \in J} \text{trans_cost}[i,j] \, x[i,j].
   \]

7.  **Formulate Constraints:**  
    - **Demand satisfaction (equality):**  
      \[
      \forall j \in J:\quad \sum_{i \in I} x[i,j] \;=\; \text{demand}[j].
      \]
    - **Linking (big‑M):**  
      \[
      \forall i \in I,\;\forall j \in J:\quad x[i,j] \;\le\; M \, y[i].
      \]
      (If supplier \(i\) is closed, no shipment can originate from it.)  
    - **Non‑negativity:**  
      \[
      \forall i \in I,\;\forall j \in J:\quad x[i,j] \;\ge\; 0.
      \]
    - **Binary opening decisions:**  
      \[
      \forall i \in I:\quad y[i] \in \{0,1\}.
      \]

[Abstract Model Plan END]