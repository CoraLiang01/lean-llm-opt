[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which suppliers to open and how to allocate their shipments to supermarkets so that every supermarket’s demand is satisfied while minimizing the sum of supplier opening (fixed) costs and per‑unit transportation costs.

2.  **Identify Model Type:** This is a **fixed‑charge transportation / uncapacitated facility location** problem, formulated as a Mixed Integer Program (MIP).

3.  **Define Index Sets:**  
   - **I** = {S1, S2, …, S12} – the set of suppliers.  
   - **J** = {C1, C2, …, C12} – the set of supermarkets (customers).

4.  **Define Decision Variables:**  
   - `x[i,j]` = quantity of goods shipped from supplier *i* to supermarket *j*.  
     Type: `GRB.CONTINUOUS`, `x[i,j] ≥ 0`.  
   - `y[i]` = 1 if supplier *i* is opened (operational), 0 otherwise.  
     Type: `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
   - **Fixed cost coefficients**: `fixed_cost[i]` from column `fixed_costs` in *fixed_cost.csv*.  
   - **Transportation cost coefficients**: `transport_cost[i,j]` from column *Cj* in row *i* of *transportation_costs.csv*.  
   - **Demand requirements**: `demand[j]` from column `demand` in *demand.csv*.  
   - **Big‑M constant**: `M` – a sufficiently large number (e.g., `M = sum_j demand[j]` or the maximum possible shipment from a supplier).

6.  **Formulate Objective:**  
   Minimize the total cost  
   \[
   \min \;\; \sum_{i \in I} \bigl( \text{fixed_cost}[i] \cdot y[i] \bigr)
          \;+\; \sum_{i \in I}\sum_{j \in J} \bigl( \text{transport_cost}[i,j] \cdot x[i,j] \bigr).
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction (for each customer j):**  
     \[
     \sum_{i \in I} x[i,j] \;\ge\; \text{demand}[j] \quad \forall j \in J.
     \]
     (Equality can be used if over‑shipment is not allowed.)  
   - **Supplier activation linking (for each supplier i and customer j):**  
     \[
     x[i,j] \;\le\; M \cdot y[i] \quad \forall i \in I,\; \forall j \in J.
     \]
     This forces all shipments from a supplier to be zero unless the supplier is opened.  
   - **Non‑negativity:**  
     \[
     x[i,j] \;\ge\; 0 \quad \forall i \in I,\; \forall j \in J.
     \]
   - **Binary supplier status:**  
     \[
     y[i] \in \{0,1\} \quad \forall i \in I.
     \]

[Abstract Model Plan END]