[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of goods should be shipped from each distribution center to each customer group so that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a classic **Transportation Problem**, which is a linear programming (LP) model with continuous decision variables.

3.  **Define Index Sets:**  
   - **S** – set of distribution centers (S1, S2, …, S12) extracted from `supply_capacity.csv`.  
   - **C** – set of customer groups (C1, C2, …, C12) extracted from `customer_demand.csv`.

4.  **Define Decision Variables:**  
   - `x[s,c]` = quantity of goods shipped from distribution center `s` to customer group `c`.  
     Type: `GRB.CONTINUOUS` (non‑negative).  

5.  **Identify Parameters (from Schema):**  
   - **Supply capacity**: `supply_capacity[s]` comes from the column `supply_capacity` in `supply_capacity.csv`.  
   - **Demand**: `demand[c]` comes from the column `demand` in `customer_demand.csv`.  
   - **Transportation cost**: `transport_cost[s,c]` comes from the cell at row `s` and column `c` in `transportation_costs.csv`.  

6.  **Formulate Objective:**  
   Minimize the total transportation cost:  
   \[
   \min \sum_{s \in S}\sum_{c \in C} \text{transport_cost}[s,c] \times x[s,c]
   \]

7.  **Formulate Constraints:**  
   - **Demand satisfaction (for each customer):**  
     \[
     \sum_{s \in S} x[s,c] = \text{demand}[c] \quad \forall c \in C
     \]
   - **Supply capacity (for each distribution center):**  
     \[
     \sum_{c \in C} x[s,c] \le \text{supply_capacity}[s] \quad \forall s \in S
     \]
   - **Non‑negativity:**  
     \[
     x[s,c] \ge 0 \quad \forall s \in S,\; \forall c \in C
     \]
[Abstract Model Plan END]