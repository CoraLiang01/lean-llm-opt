[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which suppliers to open and how to allocate the required goods from those suppliers to a set of supermarkets so that every supermarket’s demand is satisfied while minimizing the total cost, which consists of supplier opening (fixed) costs and transportation costs per unit shipped.

2.  **Identify Model Type:** This is a **Uncapacitated Facility Location Problem (UFLP)**, a mixed‑integer linear program with fixed‑charge (opening) decisions and continuous shipment decisions.

3.  **Define Index Sets:**  
   - **I** – set of suppliers (S1 … S12).  
   - **J** – set of supermarkets/customers (C1 … C12).

4.  **Define Decision Variables:**  
   - `x[i,j]` = quantity of goods shipped from supplier *i* to supermarket *j*.  
     Type: `GRB.CONTINUOUS`, `x[i,j] ≥ 0`.  
   - `y[i]` = 1 if supplier *i* is opened (operational), 0 otherwise.  
     Type: `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
   - **Fixed costs**: `fixed_cost[i]` from *fixed_cost.csv* (column “fixed_costs”).  
   - **Transportation costs**: `trans_cost[i,j]` from *transportation_costs.csv* (row *i*, column *j*).  
   - **Demand**: `demand[j]` from *demand.csv* (column “demand”).  
   - **Big‑M**: a sufficiently large constant, e.g., `M = Σ_j demand[j]` (total demand) or `M = max_j demand[j]`.

6.  **Formulate Objective:**  
   Minimize total cost  
   \[
   \min \;\; \sum_{i \in I} \text{fixed_cost}[i] \cdot y[i]
          \;+\; \sum_{i \in I}\sum_{j \in J} \text{trans_cost}[i,j] \cdot x[i,j].
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction (for each customer j):**  
     \[
     \sum_{i \in I} x[i,j] \;=\; \text{demand}[j] \quad \forall j \in J.
     \]
   - **Linking (shipment only from open suppliers):**  
     \[
     x[i,j] \;\le\; M \cdot y[i] \quad \forall i \in I,\; \forall j \in J.
     \]
     (Choosing `M = Σ_j demand[j]` guarantees the constraint is non‑binding when `y[i]=1` and forces `x[i,j]=0` when `y[i]=0`.)  
   - **Non‑negativity:**  
     \[
     x[i,j] \;\ge\; 0 \quad \forall i \in I,\; \forall j \in J.
     \]
   - **Binary supplier status:**  
     \[
     y[i] \in \{0,1\} \quad \forall i \in I.
     \]

[Abstract Model Plan END]