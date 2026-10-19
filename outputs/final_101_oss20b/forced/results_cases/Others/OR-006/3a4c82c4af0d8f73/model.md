[Abstract Model Plan START]
1. **Analyze Query:** The user wants to determine the optimal shipping quantities from each warehouse to each retail store so that all store demands are satisfied, no warehouse exceeds its daily supply capacity, and the total transportation cost is minimized.

2. **Identify Model Type:** This is a classic Linear Programming (LP) transportation problem (no integer or binary decisions are required).

3. **Define Index Sets:**  
   - **Warehouses (W):** {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10} (from *supply_capacity.csv* and *transportation_costs.csv*).  
   - **Customers (C):** {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10} (from *customer_demand.csv* and *transportation_costs.csv*).

4. **Define Decision Variables:**  
   - `x[w,c]` = quantity of product shipped from warehouse `w` to customer `c`.  
     *Type:* `GRB.CONTINUOUS` (non‑negative).

5. **Identify Parameters (from Schema):**  
   - **Demand:** `demand[c]` – the daily demand of customer `c` (column *demand* in *customer_demand.csv*).  
   - **Supply Capacity:** `supply_capacity[w]` – the daily supply capacity of warehouse `w` (column *supply_capacity* in *supply_capacity.csv*).  
   - **Transportation Cost:** `cost[w,c]` – unit transportation cost from warehouse `w` to customer `c` (row `w`, column `c` in *transportation_costs.csv*).

6. **Formulate Objective:**  
   Minimize total transportation cost:  
   \[
   \min \sum_{w \in W} \sum_{c \in C} \text{cost}[w,c] \times x[w,c]
   \]

7. **Formulate Constraints:**  
   - **Demand Satisfaction (for each customer):**  
     \[
     \sum_{w \in W} x[w,c] = \text{demand}[c] \quad \forall c \in C
     \]
   - **Supply Capacity (for each warehouse):**  
     \[
     \sum_{c \in C} x[w,c] \le \text{supply_capacity}[w] \quad \forall w \in W
     \]
   - **Non‑negativity:**  
     \[
     x[w,c] \ge 0 \quad \forall w \in W,\, c \in C
     \]
[Abstract Model Plan END]