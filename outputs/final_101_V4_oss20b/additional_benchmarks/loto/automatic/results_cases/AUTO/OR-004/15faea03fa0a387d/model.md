[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of goods should be shipped from each distribution center to each customer group so that every customer’s daily demand is satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a classic **Linear Programming (LP)** problem (continuous variables, linear objective, linear constraints).

3.  **Define Index Sets:**  
   - **S** = {S1, S2, …, S12} – the set of distribution centers (taken from the first column of *supply_capacity.csv*).  
   - **C** = {C1, C2, …, C12} – the set of customer groups (taken from the first column of *customer_demand.csv*).

4.  **Define Decision Variables:**  
   - `x[s,c]` = quantity of goods shipped from distribution center `s` to customer group `c`.  
     Type: **GRB.CONTINUOUS** (non‑negative).  
   - No binary or integer variables are required because there are no fixed‑charge or on/off decisions.

5.  **Identify Parameters (from Schema):**  
   - **Demand**: `demand[c]` – the daily demand of customer `c` (column *demand* in *customer_demand.csv*).  
   - **Supply Capacity**: `supply_capacity[s]` – the maximum quantity that distribution center `s` can ship (column *supply_capacity* in *supply_capacity.csv*).  
   - **Transportation Cost**: `cost[s,c]` – unit transportation cost from `s` to `c` (matrix formed by the columns *C1*…*C12* in *transportation_costs.csv*).

6.  **Formulate Objective:**  
   Minimize the total transportation cost:  
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
     x[s,c] \ge 0 \quad \forall s \in S,\; \forall c \in C
     \]
[Abstract Model Plan END]