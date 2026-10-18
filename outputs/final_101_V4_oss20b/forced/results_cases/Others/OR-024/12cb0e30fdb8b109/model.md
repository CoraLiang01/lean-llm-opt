[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each best‑selling product whose name starts with “S700_” should be sold in order to maximize total revenue, while respecting the available inventory and the known demand for each product.

2.  **Identify Model Type:** This is a **Linear Integer Programming (MIP)** problem (variables are integer counts of units sold, objective and constraints are linear).

3.  **Define Index Sets:**  
    - `P` : set of all products in the CSV whose `Product Name` begins with the prefix “S700_”.  
      (This set is obtained by filtering the full 109‑row table on the `Product Name` column.)

4.  **Define Decision Variables:**  
    - `x[p]` = number of units of product `p ∈ P` that the store intends to sell.  
      Type: `GRB.INTEGER` (non‑negative integer).  
    - (No additional binary variables are required for this problem.)

5.  **Identify Parameters (from Schema):**  
    - `Revenue[p]` : revenue per unit of product `p` (from the `Revenue` column).  
    - `Demand[p]`   : maximum units that customers want to buy of product `p` (from the `Demand` column).  
    - `Inv[p]`      : initial inventory available for product `p` (from the `Initial Inventory` column).

6.  **Formulate Objective:**  
    Maximize total revenue  
    \[
    \max \sum_{p \in P} \text{Revenue}[p] \; x[p]
    \]

7.  **Formulate Constraints:**  
    - **Demand limit:**  
      \[
      x[p] \;\le\; \text{Demand}[p] \quad \forall p \in P
      \]
    - **Inventory limit:**  
      \[
      x[p] \;\le\; \text{Inv}[p] \quad \forall p \in P
      \]
    - **Non‑negativity:**  
      \[
      x[p] \;\ge\; 0 \quad \forall p \in P
      \]
    (All constraints are linear and involve only the decision variables and the parameters listed above.)

[Abstract Model Plan END]