[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide which warehouses to open and how to assign the demand of each store to the warehouses such that all store demands are met, warehouse capacities are not exceeded, and the total cost (opening + transportation) is minimized.

2. **Identify Model Type:** This is a **Capacitated Facility Location Problem** (a Mixed Integer Programming model with fixed‑charge opening costs and continuous shipment variables).

3. **Define Index Sets:**  
   - **W** = {1, 2, …, 11} – set of potential warehouses (from *PotentialWarehouses_Costs.csv*).  
   - **S** = {1, 2, …, 11} – set of stores (from *Stores_Demands.csv*).

4. **Define Decision Variables:**  
   - `y[i]` = 1 if warehouse *i* is opened, 0 otherwise. **Type:** `GRB.BINARY`.  
   - `x[i,j]` = amount of goods shipped from warehouse *i* to store *j*. **Type:** `GRB.CONTINUOUS` (≥ 0).

5. **Identify Parameters (from Schema):**  
   - **Opening cost** `f[i]` – column *Opening Cost (fi)* in *PotentialWarehouses_Costs.csv*.  
   - **Capacity** `cap[i]` – column *Capacity (units)* in *PotentialWarehouses_Costs.csv*.  
   - **Demand** `d[j]` – column *Demand (units, dj)* in *Stores_Demands.csv*.  
   - **Transportation cost** `c[i,j]` – entry in *TransportationCost.csv* where the row label *W_i* and column *W_j* correspond to warehouse *i* and store *j*.

6. **Formulate Objective:**  
   Minimize  
   \[
   \sum_{i \in W} f[i]\,y[i] \;+\; \sum_{i \in W}\sum_{j \in S} c[i,j]\,x[i,j].
   \]

7. **Formulate Constraints:**  
   - **Demand satisfaction** (for each store *j*):  
     \[
     \sum_{i \in W} x[i,j] \;=\; d[j].
     \]
   - **Capacity constraint** (for each warehouse *i*):  
     \[
     \sum_{j \in S} x[i,j] \;\le\; \text{cap}[i] \; \times \; y[i].
     \]
   - **Non‑negativity** (for all *i, j*):  
     \[
     x[i,j] \;\ge\; 0.
     \]
   - **Binary opening decision** (for all *i*):  
     \[
     y[i] \;\in\; \{0,1\}.
     \]

All entities from the CSV files are used in full; no additional entities or constraints are introduced beyond those required by the problem statement.  
[Abstract Model Plan END]