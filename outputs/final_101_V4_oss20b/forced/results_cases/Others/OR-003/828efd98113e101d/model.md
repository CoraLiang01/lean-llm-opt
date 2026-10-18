[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan from multiple suppliers to customer groups, ensuring that all customer demands are met without exceeding the daily supply capacity of any supplier, while minimizing the total transportation cost.
2.  **Identify Model Type:** This is a Linear Programming (LP) transportation problem (no binary or integer decisions are required).
3.  **Define Index Sets:**  
    - **Suppliers** \(S = \{S1, S2, \dots, S10\}\) (taken from the first column of *supply_capacity.csv*).  
    - **Customers** \(C = \{C1, C2, \dots, C10\}\) (taken from the first column of *customer_demand.csv*).
4.  **Define Decision Variables:**  
    - `x[s,c]` = quantity of goods shipped from supplier `s` to customer `c`.  
      Type: `GRB.CONTINUOUS` (non‑negative).
5.  **Identify Parameters (from Schema):**  
    - **Supply capacity** `supply_capacity[s]` comes from the column `'supply_capacity'` in *supply_capacity.csv*.  
    - **Customer demand** `demand[c]` comes from the column `'demand'` in *customer_demand.csv*.  
    - **Transportation cost** `transportation_cost[s,c]` comes from the columns `C1`–`C10` in *transportation_costs.csv* (row `s`, column `c`).
6.  **Formulate Objective:**  
    Minimize total transportation cost  
    \[
    \min \sum_{s \in S}\sum_{c \in C} \text{transportation_cost}[s,c] \times x[s,c].
    \]
7.  **Formulate Constraints:**  
    - **Supply capacity constraints** (for each supplier):  
      \[
      \sum_{c \in C} x[s,c] \;\le\; \text{supply_capacity}[s], \quad \forall s \in S.
      \]
    - **Demand satisfaction constraints** (for each customer):  
      \[
      \sum_{s \in S} x[s,c] \;=\; \text{demand}[c], \quad \forall c \in C.
      \]
    - **Non‑negativity constraints:**  
      \[
      x[s,c] \;\ge\; 0, \quad \forall s \in S,\; \forall c \in C.
      \]
[Abstract Model Plan END]