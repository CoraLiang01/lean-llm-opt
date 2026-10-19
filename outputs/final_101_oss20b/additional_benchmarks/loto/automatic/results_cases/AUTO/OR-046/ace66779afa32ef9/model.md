[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide how many units of each product to order each day in order to maximize total benefit (profit) while staying within the overall storage capacity of the supermarket.

2.  **Identify Model Type:** This is a **Linear Programming (LP)** problem – all decision variables are continuous, the objective is linear, and the constraints are linear.

3.  **Define Index Sets:**  
    - **Products** – the set of all product types listed in `products.csv` (10 products in the current data set).

4.  **Define Decision Variables:**  
    - `x[p]` = number of units of product *p* to order each day.  
      Type: **GRB.CONTINUOUS** (non‑negative).  
      (If the business requires integer units, this variable could be changed to **GRB.INTEGER**.)

5.  **Identify Parameters (from Schema):**  
    - **Profit per unit** – `Value[p]` from `products.csv`.  
    - **Weight per unit** – `Weight[p]` from `products.csv`.  
    - **Total storage capacity** – `Capacity` from `capacity.csv` (single scalar).

6.  **Formulate Objective:**  
    - Maximize total daily benefit:  
      \[
      \max \sum_{p \in \text{Products}} \text{Value}[p] \times x[p]
      \]

7.  **Formulate Constraints:**  
    - **Capacity Constraint:**  
      \[
      \sum_{p \in \text{Products}} \text{Weight}[p] \times x[p] \;\le\; \text{Capacity}
      \]
    - **Non‑negativity:**  
      \[
      x[p] \;\ge\; 0 \quad \forall p \in \text{Products}
      \]
[Abstract Model Plan END]