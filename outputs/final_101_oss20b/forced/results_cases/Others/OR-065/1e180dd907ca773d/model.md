[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which distribution warehouses to activate and how to allocate inventory from those warehouses to musicians/bands so that all customer demands are satisfied while minimizing the total cost, which includes fixed opening costs for warehouses and per‑unit transportation costs.

2.  **Identify Model Type:** This is a classic **Uncapacitated Facility Location Problem (UFLP)**, a fixed‑charge location model with continuous shipment variables.

3.  **Define Index Sets:**  
   - **Facilities (I):** `{S1, S2, S3}` – the warehouses.  
   - **Customers (J):** `{C1, C2, C3}` – the musicians/bands.

4.  **Define Decision Variables:**  
   - `x[i,j]` = quantity of goods shipped from warehouse `i` to customer `j`.  
     *Type:* `GRB.CONTINUOUS`, `x[i,j] ≥ 0`.  
   - `y[i]` = 1 if warehouse `i` is opened, 0 otherwise.  
     *Type:* `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
   - **Fixed opening costs** `f[i]` come from column `fixed_costs` in `fixed_cost.csv` (matched by `Unnamed: 0` → facility ID).  
   - **Transportation costs** `c[i,j]` come from columns `C1`, `C2`, `C3` in `transportation_costs.csv` (matched by `Unnamed: 0` → facility ID).  
   - **Customer demands** `d[j]` come from column `demand` in `demand.csv` (matched by `customer` → customer ID).  
   - **Big‑M parameter** `M[i,j]` can be set to the total demand of customer `j` (or the sum of all demands) to link shipment to facility activation.

6.  **Formulate Objective:**  
   Minimize total cost  
   \[
   \min \;\; \sum_{i \in I} f[i]\,y[i] \;+\; \sum_{i \in I}\sum_{j \in J} c[i,j]\,x[i,j].
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction (for each customer):**  
     \[
     \sum_{i \in I} x[i,j] \;\ge\; d[j] \quad \forall j \in J.
     \]
     (If exact fulfillment is required, replace `≥` with `=`.)  
   - **Facility activation linking (for each shipment):**  
     \[
     x[i,j] \;\le\; M[i,j]\,y[i] \quad \forall i \in I,\; j \in J,
     \]
     where `M[i,j]` is a sufficiently large constant (e.g., `d[j]` or `∑_j d[j]`).  
   - **Non‑negativity:**  
     \[
     x[i,j] \;\ge\; 0 \quad \forall i \in I,\; j \in J.
     \]
   - **Binary constraint on facility variables:**  
     \[
     y[i] \in \{0,1\} \quad \forall i \in I.
     \]

[Abstract Model Plan END]