[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each baked good to sell in order to maximize total revenue, given that the bakery can only sell what it has in its initial inventory and must not exceed the known demand for each product.
2.  **Identify Model Type:** Based on the query, this is a **Linear Programming (LP)** problem (continuous decision variables, linear objective, linear constraints).
3.  **Define Index Sets:** The primary index set is **Products** (each row in the CSV represents a distinct product).
4.  **Define Decision Variables:**
    -   `x[p]` = quantity of product *p* that the bakery plans to sell. Type: **GRB.CONTINUOUS** (non‑negative, can be fractional if desired; otherwise integer if required).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (revenue per unit) come from column **'Revenue'**.
    -   Upper‑bound constraints on sales come from columns **'Demand'** and **'Initial Inventory'**.
6.  **Formulate Objective:**  
    Maximize total revenue:  
    \[
    \max \sum_{p \in \text{Products}} \text{Revenue}[p] \times x[p]
    \]
7.  **Formulate Constraints:**
    -   **Demand Constraint:**  
      \[
      x[p] \;\le\; \text{Demand}[p] \quad \forall p
      \]
    -   **Inventory Constraint:**  
      \[
      x[p] \;\le\; \text{Initial Inventory}[p] \quad \forall p
      \]
    -   **Non‑negativity:**  
      \[
      x[p] \;\ge\; 0 \quad \forall p
      \]
[Abstract Model Plan END]