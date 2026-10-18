[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of goods to ship from each Amazon distribution center (supplier) to each customer group so that every customer’s daily demand is satisfied, no distribution center exceeds its daily supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a classic **Transportation Problem**, a linear programming (LP) model with continuous shipment variables and no binary or integer decisions.

3.  **Define Index Sets:**  
   - **S** = {supplier1, supplier2, …, supplier8} (from *supply_capacity.csv* and the row names of *transportation_costs.csv*)  
   - **C** = {demand1, demand2, …, demand8} (from *customer_demand.csv* and the column names of *transportation_costs.csv*)

4.  **Define Decision Variables:**  
    - `x[s,c]` = quantity of goods shipped from supplier `s` to customer `c`.  
      Type: `GRB.CONTINUOUS` (non‑negative).  

5.  **Identify Parameters (from Schema):**  
    - **Transportation cost coefficients** `c[s,c]` come from the values in *transportation_costs.csv* (the cell at row `s`, column `c`).  
    - **Supply limits** `S_cap[s]` come from the `supply_capacity` column in *supply_capacity.csv*.  
    - **Demand requirements** `D_req[c]` come from the `demand` column in *customer_demand.csv*.  

6.  **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{s \in S}\sum_{c \in C} c[s,c] \cdot x[s,c]
   \]

7.  **Formulate Constraints:**  
    - **Supply constraints** (do not exceed capacity):  
      \[
      \forall s \in S:\quad \sum_{c \in C} x[s,c] \;\le\; S_{\text{cap}}[s]
      \]  
    - **Demand constraints** (fulfill each customer’s demand exactly):  
      \[
      \forall c \in C:\quad \sum_{s \in S} x[s,c] \;=\; D_{\text{req}}[c]
      \]  
    - **Non‑negativity**:  
      \[
      \forall s \in S,\;\forall c \in C:\quad x[s,c] \;\ge\; 0
      \]  
[Abstract Model Plan END]