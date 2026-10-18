[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many of each vehicle type to order each day so that the total profit from sales is maximised while the total weight of the inventory does not exceed the company’s overall capacity limit of 765 units.

2.  **Identify Model Type:** This is a **Mixed‑Integer Linear Programming (MIP)** problem (a 0‑1 knapsack‑type model with integer quantities).

3.  **Define Index Sets:**  
    - `P` : set of all vehicle types (25 products listed in *products.csv*).  
    - `p ∈ P` : individual product index.

4.  **Define Decision Variables:**  
    - `x[p]` = number of vehicles of type `p` to order per day.  
      *Type:* `GRB.INTEGER`, domain `x[p] ≥ 0`.  
    - (No binary variables are required unless a minimum‑order or on/off decision is explicitly requested.)

5.  **Identify Parameters (from Schema):**  
    - **Profit per unit**: `profit[p]` comes from the `Value` column of *products.csv*.  
    - **Weight per unit**: `weight[p]` comes from the `Weight` column of *products.csv*.  
    - **Capacity limit**: `Capacity` comes from the single value in *capacity.csv* (765).

6.  **Formulate Objective:**  
    Maximise total profit:  
    \[
    \max \;\sum_{p \in P} \text{profit}[p] \times x[p]
    \]

7.  **Formulate Constraints:**  
    - **Weight Capacity Constraint:**  
      \[
      \sum_{p \in P} \text{weight}[p] \times x[p] \;\le\; \text{Capacity}
      \]
    - **Non‑negativity & Integrality:**  
      \[
      x[p] \;\ge\; 0 \quad \text{and integer for all } p \in P
      \]
[Abstract Model Plan END]