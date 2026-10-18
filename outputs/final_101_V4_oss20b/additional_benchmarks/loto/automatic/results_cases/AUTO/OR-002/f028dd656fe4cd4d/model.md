[Abstract Model Plan START]
1. **Analyze Query:** The user wants to determine the optimal transportation plan from Walmart stores to customer groups, satisfying all customer demands without exceeding store supply capacities, while minimizing the total transportation cost.

2. **Identify Model Type:** This is a Linear Programming (LP) problem (continuous decision variables, linear objective and constraints).

3. **Define Index Sets:**  
   - **Stores (S):** `{S1, S2, …, S10}` (derived from the rows of *supply_capacity.csv* and *transportation_costs.csv*).  
   - **Customers (C):** `{C1, C2, …, C12}` (derived from the rows of *customer_demand.csv*).

4. **Define Decision Variables:**  
   - `x[s,c]` = quantity of goods transported from store `s` to customer `c`.  
     *Type:* `GRB.CONTINUOUS` (non‑negative).

5. **Identify Parameters (from Schema):**  
   - **Objective coefficients:** `cost[s,c]` from *transportation_costs.csv*.  
   - **Supply limits:** `supply_capacity[s]` from *supply_capacity.csv*.  
   - **Demand requirements:** `demand[c]` from *customer_demand.csv*.

6. **Formulate Objective:**  
   Minimize  
   \[
   \sum_{s \in S}\;\sum_{c \in C}\; cost[s,c] \times x[s,c]
   \]

7. **Formulate Constraints:**  
   - **Demand satisfaction (for each customer):**  
     \[
     \sum_{s \in S} x[s,c] = demand[c] \quad \forall\, c \in C
     \]
   - **Supply capacity (for each store):**  
     \[
     \sum_{c \in C} x[s,c] \leq supply_capacity[s] \quad \forall\, s \in S
     \]
   - **Non‑negativity:**  
     \[
     x[s,c] \geq 0 \quad \forall\, s \in S,\; c \in C
     \]
[Abstract Model Plan END]