[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of goods to ship from each Amazon distribution center to each customer group so that every customer’s daily demand is satisfied, no distribution center exceeds its daily supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a **Linear Programming (LP)** problem with continuous decision variables and a linear objective.

3.  **Define Index Sets:**  
   - **S** = {S1, S2, …, S18} – the set of distribution centers (rows in *supply_capacity.csv* and *transportation_costs.csv*).  
   - **C** = {C1, C2, …, C18} – the set of customer groups (rows in *customer_demand.csv* and columns in *transportation_costs.csv*).

4.  **Define Decision Variables:**  
    - `x[s,c]` = quantity of goods shipped from distribution center `s` to customer group `c`.  
      Type: `GRB.CONTINUOUS`, domain: `x[s,c] ≥ 0`.

5.  **Identify Parameters (from Schema):**  
    - **Demand**: `demand[c]` – daily demand of customer `c` (from *customer_demand.csv*).  
    - **Supply Capacity**: `supply_capacity[s]` – daily supply capacity of distribution center `s` (from *supply_capacity.csv*).  
    - **Transportation Cost**: `cost[s,c]` – cost per unit shipped from `s` to `c` (from *transportation_costs.csv*).  
    All entities are taken from the complete rows of each CSV file; no filtering is applied.

6.  **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{s \in S}\sum_{c \in C} \text{cost}[s,c] \times x[s,c]
   \]

7.  **Formulate Constraints:**  
   - **Demand Satisfaction (for each customer):**  
     \[
     \sum_{s \in S} x[s,c] = \text{demand}[c] \quad \forall c \in C
     \]
   - **Supply Capacity (for each distribution center):**  
     \[
     \sum_{c \in C} x[s,c] \le \text{supply_capacity}[s] \quad \forall s \in S
     \]
   - **Non‑negativity:**  
     \[
     x[s,c] \ge 0 \quad \forall s \in S,\, c \in C
     \]
[Abstract Model Plan END]