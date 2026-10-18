[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide the daily order quantity of each drug type so as to maximize total benefit while ensuring that the total weight of all ordered units does not exceed the overall inventory capacity.
2.  **Identify Model Type:** This is a Mixed Integer Linear Programming (MIP) problem – an integer knapsack formulation.
3.  **Define Index Sets:** The primary index set is **Products** (i ∈ {1,…,20}), one element for each drug type listed in *products.csv*.
4.  **Define Decision Variables:**
    - `x[i]` = number of units of drug type *i* to order daily. Type: **GRB.INTEGER**, lower bound 0.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients: `Value[i]` from the **Value** column of *products.csv*.
    - Constraint coefficients: `Weight[i]` from the **Weight** column of *products.csv*.
    - Constraint RHS: `Capacity` from the single row of *capacity.csv*.
6.  **Formulate Objective:**  
    Maximize ∑₍ᵢ₎ Value[i] · x[i]
7.  **Formulate Constraints:**
    - Capacity constraint: ∑₍ᵢ₎ Weight[i] · x[i] ≤ Capacity
    - Non‑negativity: x[i] ≥ 0 for all i
[Abstract Model Plan END]