[Abstract Model Plan START]
1. **Analyze Query:**  
   The user wants to determine the optimal daily transportation plan from Walmart stores to customer groups. The plan must satisfy all customer demands, respect each store’s daily supply capacity, and minimize the total transportation cost.

2. **Identify Model Type:**  
   This is a classic **Linear Programming (LP)** transportation problem (continuous decision variables, no integer or binary decisions required).

3. **Define Index Sets:**  
   - **Stores (S):** `{S1, S2, …, S10}` (the set of rows in *supply_capacity.csv* and *transportation_costs.csv*).  
   - **Customers (C):** `{C1, C2, …, C12}` (the set of rows in *customer_demand.csv*).

4. **Define Decision Variables:**  
   - `x[s,c]` = quantity of goods transported from store `s` to customer `c`.  
     - **Type:** `GRB.CONTINUOUS`  
     - **Domain:** `x[s,c] ≥ 0` for all `s ∈ S`, `c ∈ C`.

5. **Identify Parameters (from Schema):**  
   - **Objective coefficients:** `cost[s,c]` from *transportation_costs.csv* (column `C1`–`C12` for each store row).  
   - **Supply limits:** `supply[s]` from *supply_capacity.csv* (column `supply_capacity`).  
   - **Demand requirements:** `demand[c]` from *customer_demand.csv* (column `demand`).  
   - **Right‑hand sides:** `supply[s]` for supply constraints, `demand[c]` for demand constraints.

6. **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{s \in S}\sum_{c \in C} \text{cost}[s,c] \cdot x[s,c].
   \]

7. **Formulate Constraints:**  
   - **Demand satisfaction (equality):**  
     \[
     \forall c \in C:\quad \sum_{s \in S} x[s,c] = \text{demand}[c].
     \]
   - **Supply capacity (inequality):**  
     \[
     \forall s \in S:\quad \sum_{c \in C} x[s,c] \le \text{supply}[s].
     \]
   - **Non‑negativity:**  
     \[
     \forall s \in S,\; \forall c \in C:\quad x[s,c] \ge 0.
     \]

[Abstract Model Plan END]