[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of product to ship from each of GreenMart’s warehouses to each of its stores so that every store’s daily demand is satisfied, no warehouse exceeds its daily supply capacity, and the total transportation cost is minimized.

2.  **Identify Model Type:** This is a classic **Transportation Linear Programming (LP)** problem – a special case of a network flow model with only supply, demand, and cost parameters.

3.  **Define Index Sets:**  
   - **Warehouses (W):** `{S1, S2, S3, S4, S5}` – taken from the rows of *supply_capacity.csv* (column `region`).  
   - **Stores (C):** `{D1, D2, D3, D4, D5}` – taken from the rows of *customer_demand.csv* (column `customer`) and the columns of *transportation_costs.csv* (excluding the first column `Unnamed: 0`).

4.  **Define Decision Variables:**  
   - `x[w,c]` = quantity of product shipped from warehouse `w` to store `c`.  
     *Type:* `GRB.CONTINUOUS` (non‑negative).  
   - No binary or integer variables are required for this problem.

5.  **Identify Parameters (from Schema):**  
   - **Demand parameters** `d[c]` come from *customer_demand.csv* column `demand`.  
   - **Supply parameters** `s[w]` come from *supply_capacity.csv* column `supply_capacity`.  
   - **Transportation cost parameters** `c[w,c]` come from *transportation_costs.csv* – the value in row `w` and column `c`.  
   - All indices are taken from the complete entity sets; no filtering is applied.

6.  **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{w \in W}\sum_{c \in C} c[w,c] \; x[w,c]
   \]

7.  **Formulate Constraints:**  
   - **Supply constraints (warehouse capacity):**  
     \[
     \forall w \in W:\quad \sum_{c \in C} x[w,c] \;\le\; s[w]
     \]
   - **Demand constraints (store requirement):**  
     \[
     \forall c \in C:\quad \sum_{w \in W} x[w,c] \;=\; d[c]
     \]
   - **Non‑negativity:**  
     \[
     \forall w \in W,\;\forall c \in C:\quad x[w,c] \;\ge\; 0
     \]

[Abstract Model Plan END]