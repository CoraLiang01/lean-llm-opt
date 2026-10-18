[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to choose a subset of warehouses to open and assign all customer demand to them, minimizing the sum of fixed opening costs and variable transportation costs while respecting each warehouse’s capacity and meeting every customer’s demand exactly.
2.  **Identify Model Type:** This is a Capacitated Facility Location Problem (CFLP), a Mixed Integer Linear Programming (MILP) model with fixed‑charge (binary) and continuous shipment variables.
3.  **Define Index Sets:**  
    - **W** = {W1, W2, …, W10} – the set of potential warehouses (rows of *warehouse.csv*).  
    - **C** = {C1, C2, …, C20} – the set of customers (rows of *demand.csv* and columns of *cost.csv*).
4.  **Define Decision Variables:**  
    - `x[w,c]` = quantity of goods shipped from warehouse `w` to customer `c`.  
      Type: `GRB.CONTINUOUS`, with `x[w,c] ≥ 0`.  
    - `y[w]` = 1 if warehouse `w` is opened, 0 otherwise.  
      Type: `GRB.BINARY`.
5.  **Identify Parameters (from Schema):**  
    - **Fixed opening cost** `f[w]` – from *warehouse.csv* column **Fixed_Cost**.  
    - **Capacity** `cap[w]` – from *warehouse.csv* column **Capacity**.  
    - **Demand** `d[c]` – from *demand.csv* column **Demand**.  
    - **Transportation cost** `t[w,c]` – from *cost.csv* columns **C1** … **C20** (matching customer IDs).
6.  **Formulate Objective:**  
    Minimize  
    \[
    \sum_{w \in W} f[w] \, y[w] \;+\; \sum_{w \in W}\sum_{c \in C} t[w,c] \, x[w,c].
    \]
7.  **Formulate Constraints:**  
    - **Demand satisfaction** (for each customer):  
      \[
      \sum_{w \in W} x[w,c] \;=\; d[c] \quad \forall c \in C.
      \]
    - **Warehouse capacity** (for each warehouse):  
      \[
      \sum_{c \in C} x[w,c] \;\le\; cap[w] \, y[w] \quad \forall w \in W.
      \]
    - **Variable bounds**:  
      \[
      x[w,c] \;\ge\; 0 \quad \forall w \in W,\, c \in C, \qquad
      y[w] \in \{0,1\} \quad \forall w \in W.
      \]
[Abstract Model Plan END]