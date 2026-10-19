[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine an integer production plan for 80 products over a 22‑day horizon that maximizes total profit, taking into account revenue, production costs, fixed activation costs, production quotas, maximum demand, and minimum batch size constraints.
2.  **Identify Model Type:** This is a Mixed Integer Linear Programming (MILP) problem with fixed‑charge (activation) costs – a classic fixed‑charge production planning model.
3.  **Define Index Sets:** 
   - `P` = {1, 2, …, 80} – the set of products.
4.  **Define Decision Variables:**
    - `x[p]` = number of 100‑kg units of product *p* produced. Type: integer, `x[p] ≥ 0`.
    - `y[p]` = 1 if product *p* is activated (i.e., at least one unit is produced), 0 otherwise. Type: binary.
5.  **Identify Parameters (from Schema):**
    - `Demand[p]` – maximum demand for product *p* (from column A1–A80 in 36‑1.csv).
    - `Price[p]` – selling price per 100‑kg unit of product *p* (from column A1–A80 in 36‑1.csv).
    - `Cost[p]` – production cost per 100‑kg unit of product *p* (from column A1–A80 in 36‑1.csv).
    - `Quota[p]` – maximum number of 100‑kg units that can be produced per day if all production lines are dedicated to product *p* (from column A1–A80 in 36‑1.csv).
    - `FixedCost[p]` – fixed activation cost for product *p* (from column A1–A80 in 36‑2.csv).
    - `MinBatch[p]` – minimum batch size (in 100‑kg units) for product *p* (from column A1–A80 in 36‑3.csv).
6.  **Formulate Objective:**  
   Maximize total profit  
   \[
   \max \sum_{p\in P} \bigl( (Price[p] - Cost[p]) \cdot x[p] \;-\; FixedCost[p] \cdot y[p] \bigr)
   \]
7.  **Formulate Constraints:**
    - **Demand limit:**  
      \[
      x[p] \;\le\; Demand[p] \quad \forall p\in P
      \]
    - **Daily quota limit (over 22 days):**  
      \[
      x[p] \;\le\; Quota[p] \times 22 \quad \forall p\in P
      \]
    - **Activation linking (production only if activated):**  
      \[
      x[p] \;\le\; Demand[p] \times y[p] \quad \forall p\in P
      \]
      \[
      x[p] \;\le\; Quota[p] \times 22 \times y[p] \quad \forall p\in P
      \]
    - **Minimum batch size (if activated):**  
      \[
      x[p] \;\ge\; MinBatch[p] \times y[p] \quad \forall p\in P
      \]
    - **Shared production‑day resource constraint:**  
      \[
      \sum_{p\in P} \frac{x[p]}{Quota[p]} \;\le\; 22
      \]
    - **Variable domains:**  
      \[
      x[p] \in \mathbb{Z}_{\ge 0}, \quad y[p] \in \{0,1\} \quad \forall p\in P
      \]
[Abstract Model Plan END]