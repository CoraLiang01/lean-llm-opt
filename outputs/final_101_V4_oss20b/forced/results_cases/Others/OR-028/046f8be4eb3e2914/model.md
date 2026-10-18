[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many units of each product to sell in order to maximize total revenue, while respecting the known demand for each product and the initial inventory available at the start of the sales horizon.  
2. **Identify Model Type:** This is a **Linear Integer Programming (MIP)** problem (variables are integer counts of units sold).  
3. **Define Index Sets:**  
   - **Products** \(i \in I\) – each row in the CSV represents a distinct product.  
4. **Define Decision Variables:**  
   - `x[i]` = number of units of product *i* that will be fulfilled (sold).  
     - **Type:** `GRB.INTEGER` (non‑negative integer).  
5. **Identify Parameters (from Schema):**  
   - **Revenue[i]** – revenue per unit of product *i*, taken from the **'Revenue'** column.  
   - **Demand[i]** – maximum customer demand for product *i*, taken from the **'Demand'** column.  
   - **InitialInventory[i]** – units of product *i* available at the start of the horizon, taken from the **'Initial Inventory'** column.  
6. **Formulate Objective:**  
   - **Maximize** \(\displaystyle \sum_{i \in I} \text{Revenue}[i] \times x[i]\).  
7. **Formulate Constraints:**  
   - **Demand Constraint:**  
     \[
     x[i] \;\le\; \text{Demand}[i] \quad \forall i \in I
     \]  
   - **Inventory Constraint:**  
     \[
     x[i] \;\le\; \text{InitialInventory}[i] \quad \forall i \in I
     \]  
   - **Non‑negativity:**  
     \[
     x[i] \;\ge\; 0 \quad \forall i \in I
     \]  
   (All constraints are linear and involve only the decision variables and the parameters extracted from the CSV.)  
[Abstract Model Plan END]