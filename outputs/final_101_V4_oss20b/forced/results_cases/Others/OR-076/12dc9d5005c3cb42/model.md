[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to choose a subset of potential warehouse locations to open and assign all customer demand to the opened warehouses, minimizing the sum of fixed opening costs and variable transportation costs while respecting each warehouse’s maximum service capacity.

2.  **Identify Model Type:** This is a Capacitated Facility Location Problem (a Mixed Integer Linear Program with binary facility‑opening variables and continuous assignment variables).

3.  **Define Index Sets:**  
    - **W** = {W1, W2, …, W10}  (set of warehouses)  
    - **C** = {C1, C2, …, C20}  (set of customers)

4.  **Define Decision Variables:**  
    - `x[w,c]` = amount of customer c’s demand served by warehouse w.  
      *Type:* `GRB.CONTINUOUS`, `x[w,c] ≥ 0`.  
    - `y[w]` = 1 if warehouse w is opened, 0 otherwise.  
      *Type:* `GRB.BINARY`.

5.  **Identify Parameters (from Schema):**  
    - **Fixed cost:** `fixed_cost[w]` from `warehouse.csv` column *Fixed_Cost*.  
    - **Capacity:** `capacity[w]` from `warehouse.csv` column *Capacity*.  
    - **Demand:** `demand[c]` from `demand.csv` column *Demand*.  
    - **Transportation cost:** `cost[w,c]` from `cost.csv` columns *C1*–*C20* (row w, column c).

6.  **Formulate Objective:**  
    Minimize  
    \[
    \sum_{w\in W} \text{fixed_cost}[w] \cdot y[w]
    \;+\;
    \sum_{w\in W}\sum_{c\in C} \text{cost}[w,c] \cdot x[w,c].
    \]

7.  **Formulate Constraints:**  
    - **Demand satisfaction:**  
      \[
      \forall c\in C:\quad \sum_{w\in W} x[w,c] = \text{demand}[c].
      \]
    - **Capacity constraint:**  
      \[
      \forall w\in W:\quad \sum_{c\in C} x[w,c] \le \text{capacity}[w] \cdot y[w].
      \]
    - **Non‑negativity:**  
      \[
      \forall w\in W,\;\forall c\in C:\quad x[w,c] \ge 0.
      \]
    - **Binary opening decision:**  
      \[
      \forall w\in W:\quad y[w] \in \{0,1\}.
      \]
[Abstract Model Plan END]