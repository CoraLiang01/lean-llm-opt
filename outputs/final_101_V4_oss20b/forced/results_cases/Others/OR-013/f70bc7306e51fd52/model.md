[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each “4U” product to sell in order to maximize total revenue, subject to the limits imposed by the available inventory and the known demand for each product.  
2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem (all decision variables are integers, the objective and constraints are linear).  
3.  **Define Index Sets:**  
    - **I** – set of all products whose *Product Name* contains the substring “4U” (case‑insensitive).  
4.  **Define Decision Variables:**  
    - `x[i]` = number of units of product *i* that the retailer plans to sell.  
      *Type:* `GRB.INTEGER` (non‑negative integer).  
5.  **Identify Parameters (from Schema):**  
    - `Revenue[i]` – revenue per unit of product *i* (from column **Revenue**).  
    - `Demand[i]` – maximum demand for product *i* (from column **Demand**).  
    - `Inv[i]` – initial inventory available for product *i* (from column **Initial Inventory**).  
    - `Cap[i] = min(Demand[i], Inv[i])` – the upper bound on sales for product *i*.  
6.  **Formulate Objective:**  
    - **Maximize**  
      \[
      \sum_{i \in I} \text{Revenue}[i] \times x[i]
      \]  
      (total revenue from all sold units).  
7.  **Formulate Constraints:**  
    - **Inventory / Demand Limits** (for each product *i*):  
      \[
      0 \;\le\; x[i] \;\le\; \text{Cap}[i] \quad \forall i \in I
      \]  
      (sales cannot exceed the lesser of available inventory or realized demand).  
    - **Integer Requirement** (already encoded in variable type).  
[Abstract Model Plan END]