[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many units to ship from each Amazon distribution center (supplier) to each customer group (customer) so that all customer demands are exactly met, no distribution center exceeds its daily supply capacity, and the total transportation cost is minimized.

2. **Identify Model Type:** This is a **Linear Programming (LP)** problem with continuous decision variables and no binary or integer variables.

3. **Define Index Sets:**  
   - **Suppliers** \(S = \{\text{supply1}, \text{supply2}, \dots, \text{supply8}\}\)  
   - **Customers** \(C = \{\text{demand1}, \text{demand2}, \dots, \text{demand8}\}\)

4. **Define Decision Variables:**  
   - `x[s,c]` = quantity shipped from supplier `s` to customer `c`.  
     *Type:* `GRB.CONTINUOUS` (non‑negative).

5. **Identify Parameters (from Schema):**  
   - `demand[c]` – daily demand for customer `c` (from *customer_demand.csv*).  
   - `supply_capacity[s]` – daily supply capacity of supplier `s` (from *supply_capacity.csv*).  
   - `transport_cost[s,c]` – transportation cost per unit from supplier `s` to customer `c` (from *transportation_costs.csv*).

6. **Formulate Objective:**  
   Minimize total transportation cost  
   \[
   \min \sum_{s \in S}\sum_{c \in C} \text{transport_cost}[s,c] \times x[s,c]
   \]

7. **Formulate Constraints:**  
   - **Demand fulfillment (equality):**  
     \[
     \forall c \in C:\quad \sum_{s \in S} x[s,c] = \text{demand}[c]
     \]
   - **Supply capacity (inequality):**  
     \[
     \forall s \in S:\quad \sum_{c \in C} x[s,c] \le \text{supply_capacity}[s]
     \]
   - **Non‑negativity:**  
     \[
     \forall s \in S,\; \forall c \in C:\quad x[s,c] \ge 0
     \]
[Abstract Model Plan END]