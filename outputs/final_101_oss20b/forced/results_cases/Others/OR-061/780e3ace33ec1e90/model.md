[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which suppliers to open and how to allocate shipments from those suppliers to each branch so that every branch’s demand is satisfied while minimizing the sum of supplier fixed opening costs and per‑unit transportation costs.

2.  **Identify Model Type:** This is a **Fixed‑Charge Transportation Problem** (a Mixed Integer Linear Program with binary opening variables and continuous shipment variables).

3.  **Define Index Sets:**  
   - **Suppliers** \(I = \{S1, S2, S3, S4, S5\}\)  
   - **Branches (Customers)** \(J = \{C1, C2, C3, C4, C5\}\)

4.  **Define Decision Variables:**  
   - `x[i,j]` = quantity of goods shipped from supplier `i` to branch `j`.  
     Type: `GRB.CONTINUOUS`, \(x[i,j] \ge 0\).  
   - `y[i]` = 1 if supplier `i` is opened (operational), 0 otherwise.  
     Type: `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
   - **Fixed opening costs**: `fixed_cost[i]` from *fixed_cost.csv* (column `fixed_costs`).  
   - **Transportation costs per unit**: `trans_cost[i,j]` from *transportation_costs.csv* (columns `C1`–`C5`).  
   - **Branch demands**: `demand[j]` from *demand.csv* (column `demand`).  
   - **Big‑M constant**: `M = sum_j demand[j] = 187` (upper bound on any shipment from a supplier).

6.  **Formulate Objective:**  
   Minimize  
   \[
   \sum_{i \in I} \text{fixed_cost}[i] \cdot y[i]
   \;+\;
   \sum_{i \in I}\sum_{j \in J} \text{trans_cost}[i,j] \cdot x[i,j].
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction** (for each branch \(j\)):  
     \[
     \sum_{i \in I} x[i,j] \;=\; \text{demand}[j].
     \]
   - **Supplier opening linkage** (for each supplier \(i\) and branch \(j\)):  
     \[
     x[i,j] \;\le\; M \cdot y[i].
     \]
     (ensures no shipment can be sent from a closed supplier).  
   - **Non‑negativity**:  
     \[
     x[i,j] \;\ge\; 0 \quad \forall i \in I,\, j \in J.
     \]
   - **Binary opening variables**:  
     \[
     y[i] \;\in\; \{0,1\} \quad \forall i \in I.
     \]
[Abstract Model Plan END]