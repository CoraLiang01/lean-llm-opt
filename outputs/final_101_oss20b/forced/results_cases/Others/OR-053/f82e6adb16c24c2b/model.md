[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each product should be placed on each shelf so that the total value of all products on all shelves is maximized, while ensuring that the total weight of products on any given shelf does not exceed that shelf’s capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is a pure Integer Linear Programming (ILP) problem (a multi‑knapsack assignment without binary activation variables).

3.  **Define Index Sets:**  
    - **S** = {1, 2, …, 10} – the set of shelf identifiers (from `capacity.csv`).  
    - **P** = {1, 2, …, 20} – the set of product identifiers (from `products.csv`).

4.  **Define Decision Variables:**  
    - `x[s,p]` = number of units of product *p* placed on shelf *s*.  
      Type: **GRB.INTEGER** (non‑negative integers).  
      Domain: `x[s,p] ≥ 0` for all *s* ∈ S, *p* ∈ P.

5.  **Identify Parameters (from Schema):**  
    - **Value[p]** – the monetary value of one unit of product *p* (column `Value` in `products.csv`).  
    - **Weight[p]** – the weight of one unit of product *p* (column `Weight` in `products.csv`).  
    - **Capacity[s]** – the maximum total weight that shelf *s* can hold (column `Capacity` in `capacity.csv`).

6.  **Formulate Objective:**  
    Maximize the total value of all products placed on all shelves:  
    \[
    \max \sum_{s \in S} \sum_{p \in P} \text{Value}[p] \times x[s,p]
    \]

7.  **Formulate Constraints:**  
    - **Shelf Capacity Constraints:** For each shelf *s* ∈ S, the total weight of products on that shelf must not exceed its capacity:  
      \[
      \sum_{p \in P} \text{Weight}[p] \times x[s,p] \;\le\; \text{Capacity}[s] \quad \forall s \in S
      \]
    - **Non‑negativity and Integer Constraints:**  
      \[
      x[s,p] \;\ge\; 0 \quad \text{and integer} \quad \forall s \in S,\, p \in P
      \]

[Abstract Model Plan END]