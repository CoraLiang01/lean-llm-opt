[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide which warehouses to open and how to allocate each store’s demand to the opened warehouses so that all store demands are satisfied, no warehouse exceeds its capacity, and the total cost (sum of opening costs and transportation costs) is minimized.

2. **Identify Model Type:** This is a **Capacitated Facility Location Problem** (a fixed‑charge mixed‑integer program).

3. **Define Index Sets:**  
   - \(I = \{1,2,\dots,11\}\) – set of potential warehouses (from *PotentialWarehouses_Costs.csv*).  
   - \(J = \{1,2,\dots,11\}\) – set of stores (from *Stores_Demands.csv*).

4. **Define Decision Variables:**  
   - `y[i]` = 1 if warehouse \(i\) is opened, 0 otherwise. Type: **GRB.BINARY**.  
   - `x[i,j]` = units shipped from warehouse \(i\) to store \(j\). Type: **GRB.CONTINUOUS** (≥ 0).

5. **Identify Parameters (from Schema):**  
   - **Opening cost** \(f[i]\) – from column *Opening Cost (fi)* in *PotentialWarehouses_Costs.csv*.  
   - **Capacity** \(cap[i]\) – from column *Capacity (units)* in *PotentialWarehouses_Costs.csv*.  
   - **Demand** \(d[j]\) – from column *Demand (units, dj)* in *Stores_Demands.csv*.  
   - **Transportation cost** \(c[i,j]\) – from *TransportationCost.csv*: the entry in row “Wj” and column “Wi” gives the cost of shipping one unit from warehouse \(i\) to store \(j\).

6. **Formulate Objective:**  
   Minimize  
   \[
   \sum_{i\in I} f[i]\;y[i] \;+\; \sum_{i\in I}\sum_{j\in J} c[i,j]\;x[i,j].
   \]

7. **Formulate Constraints:**  
   - **Demand satisfaction** (for each store \(j\)):  
     \[
     \sum_{i\in I} x[i,j] \;=\; d[j].
     \]
   - **Capacity linking** (for each warehouse \(i\)):  
     \[
     \sum_{j\in J} x[i,j] \;\le\; cap[i]\;y[i].
     \]
   - **Non‑negativity** (for all \(i,j\)):  
     \[
     x[i,j] \;\ge\; 0.
     \]
   - **Binary opening decision** (for all \(i\)):  
     \[
     y[i] \;\in\; \{0,1\}.
     \]

[Abstract Model Plan END]