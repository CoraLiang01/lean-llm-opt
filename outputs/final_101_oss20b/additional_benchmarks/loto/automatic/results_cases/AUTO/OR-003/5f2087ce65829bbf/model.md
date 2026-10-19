[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of goods should be shipped from each supplier to each customer group so that all customer demands are satisfied, no supplier exceeds its daily supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a **Linear Programming (LP)** problem: all decision variables are continuous, the objective is linear, and all constraints are linear.

3.  **Define Index Sets:**  
   - **Suppliers** \(S = \{S1, S2, \dots, S10\}\) (taken from the first column of *supply_capacity.csv*).  
   - **Customers** \(C = \{C1, C2, \dots, C10\}\) (taken from the first column of *customer_demand.csv*).

4.  **Define Decision Variables:**  
   - `x[s,c]` = quantity of goods shipped from supplier `s` to customer `c`.  
     Type: `GRB.CONTINUOUS`, \(x[s,c] \ge 0\).

5.  **Identify Parameters (from Schema):**  
   - **Supply capacity**: `cap[s]` from *supply_capacity.csv* (column `supply_capacity`).  
   - **Customer demand**: `dem[c]` from *customer_demand.csv* (column `demand`).  
   - **Transportation cost**: `cost[s,c]` from *transportation_costs.csv* (columns `C1`–`C10`).

6.  **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{s \in S}\sum_{c \in C} \text{cost}[s,c] \cdot x[s,c].
   \]

7.  **Formulate Constraints:**  
   - **Supply capacity constraints** (for each supplier):  
     \[
     \sum_{c \in C} x[s,c] \;\le\; \text{cap}[s], \quad \forall s \in S.
     \]
   - **Demand satisfaction constraints** (for each customer):  
     \[
     \sum_{s \in S} x[s,c] \;=\; \text{dem}[c], \quad \forall c \in C.
     \]
   - **Non‑negativity**:  
     \[
     x[s,c] \;\ge\; 0, \quad \forall s \in S,\; \forall c \in C.
     \]

[Abstract Model Plan END]