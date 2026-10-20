[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine an integer daily ordering plan for each vehicle type that maximizes the total benefit (profit) while respecting two types of capacity limits: a per‑vehicle‑type daily inventory limit and an overall daily inventory capacity that cannot be exceeded.

2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem (all variables are integers, the objective and constraints are linear).

3.  **Define Index Sets:**  
    - `V` : set of vehicle types (one element per row in the CSV files, e.g., Sedans, SUVs, Electric Vehicles, …).

4.  **Define Decision Variables:**  
    - `x[v]` = number of vehicles of type `v` to order on a given day.  
      Type: `GRB.INTEGER` (non‑negative).  
    - (No binary or continuous auxiliary variables are required for this formulation.)

5.  **Identify Parameters (from Schema):**  
    - **Benefit coefficient** (profit per unit) comes from `products.csv` column `Value`.  
      Parameter: `benefit[v]`.  
    - **Per‑type daily capacity** comes from `capacity.csv` column `Capacity`.  
      Parameter: `cap[v]`.  
    - **Total daily capacity** is the sum of all per‑type capacities:  
      `total_cap = Σ_v cap[v]`.  
      (This can be pre‑computed or expressed as a separate parameter.)

6.  **Formulate Objective:**  
    Maximize total benefit:  
    \[
    \max \;\; \sum_{v \in V} \text{benefit}[v] \times x[v]
    \]

7.  **Formulate Constraints:**  
    - **Per‑type capacity limits:**  
      \[
      x[v] \;\le\; \text{cap}[v] \quad \forall v \in V
      \]  
    - **Global inventory capacity:**  
      \[
      \sum_{v \in V} x[v] \;\le\; \text{total\_cap}
      \]  
    - **Non‑negativity and integrality:**  
      \[
      x[v] \;\ge\; 0 \quad \text{and integer} \quad \forall v \in V
      \]  
    (The `Weight` column in `products.csv` is not used in this model unless a secondary objective or constraint is added.)

[Abstract Model Plan END]