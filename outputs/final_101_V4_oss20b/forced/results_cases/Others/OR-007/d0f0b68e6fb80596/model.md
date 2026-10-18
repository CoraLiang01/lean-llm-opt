[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities from each warehouse to each store so that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a classic **Linear Programming (LP)** transportation problem.

3.  **Define Index Sets:**  
   - **S** = {S1, S2, S3, S4, S5}  (warehouses)  
   - **D** = {D1, D2, D3, D4, D5}  (stores)

4.  **Define Decision Variables:**  
    - `x[s,d]` = quantity of product shipped from warehouse `s` to store `d`.  
      Type: **GRB.CONTINUOUS** (non‑negative).

5.  **Identify Parameters (from Schema):**  
    - **Objective coefficients**: `cost[s,d]` from *transportation_costs.csv* (column `D1`‑`D5` for each row `S1`‑`S5`).  
    - **Supply limits**: `supply[s]` from *supply_capacity.csv* (`supply_capacity` column).  
    - **Demand requirements**: `demand[d]` from *customer_demand.csv* (`demand` column).

6.  **Formulate Objective:**  
   Minimize  
   \[
   \sum_{s \in S}\;\sum_{d \in D}\; cost[s,d] \times x[s,d]
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction (equality):**  
     \[
     \forall d \in D:\;\;\sum_{s \in S} x[s,d] = demand[d]
     \]
   - **Supply capacity (inequality):**  
     \[
     \forall s \in S:\;\;\sum_{d \in D} x[s,d] \le supply[s]
     \]
   - **Non‑negativity:**  
     \[
     \forall s \in S,\;\forall d \in D:\;\;x[s,d] \ge 0
     \]
[Abstract Model Plan END]