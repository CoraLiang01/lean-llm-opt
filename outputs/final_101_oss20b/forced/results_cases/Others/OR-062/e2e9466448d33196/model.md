[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide which liquor suppliers to activate and how many units each licensed store should source from each supplier so that every store’s demand is satisfied while minimizing the total cost, which includes supplier fixed opening costs and per‑unit transportation costs.

2. **Identify Model Type:** This is a **Fixed‑Charge Uncapacitated Facility Location Problem (UFLP)**, a Mixed Integer Programming (MIP) model with binary facility‑opening variables and continuous shipment variables.

3. **Define Index Sets:**  
   - **I** – set of suppliers (facilities):  
     `{MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES}`  
   - **J** – set of stores (customers):  
     `{CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT}`  

4. **Define Decision Variables:**  
   - `x[i,j]` = quantity of liquor units shipped from supplier `i` to store `j`.  
     *Type:* `GRB.CONTINUOUS`, lower bound `0`.  
   - `y[i]` = 1 if supplier `i` is opened (operational), 0 otherwise.  
     *Type:* `GRB.BINARY`.

5. **Identify Parameters (from Schema):**  
   - `fixed_cost[i]` – supplier opening cost, taken from the column **fixed_costs** in *fixed_cost.csv*.  
   - `transport_cost[i,j]` – per‑unit transportation cost from supplier `i` to store `j`, taken from the corresponding column `j` in *transportation_costs.csv*.  
   - `demand[j]` – required quantity for store `j`, taken from the column **demand** in *demand.csv*.

6. **Formulate Objective:**  
   Minimize total cost  
   \[
   \min \;\; \sum_{i\in I} \text{fixed_cost}[i] \cdot y[i]
          \;+\; \sum_{i\in I}\sum_{j\in J} \text{transport_cost}[i,j] \cdot x[i,j].
   \]

7. **Formulate Constraints:**  
   - **Demand satisfaction** (for each store):  
     \[
     \sum_{i\in I} x[i,j] \;=\; \text{demand}[j] \quad \forall j\in J.
     \]
   - **Facility‑linking** (ensures shipments only from opened suppliers):  
     \[
     x[i,j] \;\le\; M \cdot y[i] \quad \forall i\in I,\; \forall j\in J,
     \]
     where `M` is a sufficiently large constant (e.g., the sum of all store demands).  
   - **Non‑negativity**:  
     \[
     x[i,j] \;\ge\; 0 \quad \forall i\in I,\; \forall j\in J.
     \]
   - **Binary**:  
     \[
     y[i] \;\in\; \{0,1\} \quad \forall i\in I.
     \]

[Abstract Model Plan END]