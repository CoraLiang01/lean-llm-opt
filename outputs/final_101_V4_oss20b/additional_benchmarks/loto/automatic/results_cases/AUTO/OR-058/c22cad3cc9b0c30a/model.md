[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide which suppliers to open and how much to ship from each supplier to each store so that all store demands are met while minimizing the total cost, which includes supplier opening (fixed) costs and per‑unit transportation costs.
2.  **Identify Model Type:** This is a Uncapacitated Facility Location (UFL) problem, a type of Fixed‑Charge Mixed Integer Programming (MIP) model.
3.  **Define Index Sets:**  
    - Suppliers: \(I = \{S1, S2, S3, S4, S5, S6\}\)  
    - Stores: \(J = \{C1, C2, C3, C4, C5, C6\}\)
4.  **Define Decision Variables:**  
    - `x[i,j]` = quantity of product shipped from supplier `i` to store `j`. Type: `GRB.CONTINUOUS` (non‑negative).  
    - `y[i]` = 1 if supplier `i` is opened, 0 otherwise. Type: `GRB.BINARY`.
5.  **Identify Parameters (from Schema):**  
    - `fixed_cost[i]` from column `'fixed_costs'` in *fixed_cost.csv*.  
    - `transportation_cost[i,j]` from column `j` in *transportation_costs.csv* for row `i`.  
    - `demand[j]` from column `'demand'` in *demand.csv*.  
    - `M` (big‑M) = sum of all store demands (or a sufficiently large constant).
6.  **Formulate Objective:**  
    Minimize  
    \[
    \sum_{i \in I} \text{fixed_cost}[i] \cdot y[i] \;+\; \sum_{i \in I}\sum_{j \in J} \text{transportation_cost}[i,j] \cdot x[i,j]
    \]
7.  **Formulate Constraints:**  
    - **Demand satisfaction:** For each store \(j\),  
      \[
      \sum_{i \in I} x[i,j] \;\ge\; \text{demand}[j] \quad (\text{or } = \text{demand}[j] \text{ if exact fulfillment is required})
      \]
    - **Linking (open‑ship) constraint:** For each supplier \(i\) and store \(j\),  
      \[
      x[i,j] \;\le\; M \cdot y[i]
      \]
    - **Non‑negativity:** \(x[i,j] \ge 0\) for all \(i,j\).  
    - **Binary:** \(y[i] \in \{0,1\}\) for all \(i\).
[Abstract Model Plan END]