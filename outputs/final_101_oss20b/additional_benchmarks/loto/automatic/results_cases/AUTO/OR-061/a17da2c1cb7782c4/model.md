[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which suppliers to activate and how to allocate shipments to meet branch demands at minimum total cost, including fixed supplier activation costs and per‑unit transportation costs.
2.  **Identify Model Type:** This is a fixed‑charge location problem (Uncapacitated Facility Location Problem) with continuous shipment variables.
3.  **Define Index Sets:** The primary indices are  
   - Suppliers `I = {S1, S2, S3, S4, S5}`  
   - Customers `J = {C1, C2, C3, C4, C5}`.
4.  **Define Decision Variables:**  
    - `x[i,j]` = quantity of goods shipped from supplier `i` to customer `j`. Type: `GRB.CONTINUOUS`.  
    - `y[i]` = 1 if supplier `i` is opened, 0 otherwise. Type: `GRB.BINARY`.
5.  **Identify Parameters (from Schema):**  
    - Fixed costs: `fixed_cost[i]` from column `'fixed_costs'` in `fixed_cost.csv`.  
    - Transportation costs: `trans_cost[i,j]` from column `j` in `transportation_costs.csv`.  
    - Demand: `demand[j]` from column `'demand'` in `demand.csv`.  
    - Big‑M: `M` can be set to the sum of all demands or a sufficiently large number.
6.  **Formulate Objective:**  
    Minimize total cost  
    \[
    \min \sum_{i\in I} \text{fixed_cost}[i] \cdot y[i] \;+\; \sum_{i\in I}\sum_{j\in J} \text{trans_cost}[i,j] \cdot x[i,j].
    \]
7.  **Formulate Constraints:**  
    - **Demand satisfaction:**  
      \[
      \forall j\in J:\;\; \sum_{i\in I} x[i,j] \;=\; \text{demand}[j].
      \]
    - **Linking (no shipment from closed supplier):**  
      \[
      \forall i\in I,\;\forall j\in J:\;\; x[i,j] \;\le\; M \cdot y[i].
      \]
    - **Non‑negativity:**  
      \[
      \forall i\in I,\;\forall j\in J:\;\; x[i,j] \;\ge\; 0.
      \]
    - **Binary supplier activation:**  
      \[
      \forall i\in I:\;\; y[i] \in \{0,1\}.
      \]
[Abstract Model Plan END]