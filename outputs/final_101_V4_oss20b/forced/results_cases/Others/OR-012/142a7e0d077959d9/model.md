[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to maximize total revenue by deciding how many units of each product to sell, subject to deterministic demand and the retailer’s initial inventory limits.
2.  **Identify Model Type:** This is a Linear Programming / Integer Programming (LP/IP) resource‑allocation problem.
3.  **Define Index Sets:**  
    - `I` : set of all products (indexed by product name).
4.  **Define Decision Variables:**  
    - `x[i]` = number of units of product *i* sold.  
      Type: `GRB.INTEGER` (use `GRB.CONTINUOUS` if fractional sales were allowed).
5.  **Identify Parameters (from Schema):**  
    - `Revenue[i]` – revenue per unit, from column **Revenue**.  
    - `Demand[i]` – maximum units that can be sold, from column **Demand**.  
    - `Inv[i]` – initial inventory available, from column **Initial Inventory**.
6.  **Formulate Objective:**  
    - `Maximize  Σ_{i∈I} Revenue[i] * x[i]`
7.  **Formulate Constraints:**  
    - **Demand limit:** ∀ i ∈ I `x[i] ≤ Demand[i]`  
    - **Inventory limit:** ∀ i ∈ I `x[i] ≤ Inv[i]`  
    - **Non‑negativity:** ∀ i ∈ I `x[i] ≥ 0`  
    *(The two upper bounds can be combined as `x[i] ≤ min(Demand[i], Inv[i])` if desired.)*
[Abstract Model Plan END]