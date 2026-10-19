[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate products to displays so as to maximize the total value of all products placed, subject to each display’s weight capacity and a minimum total quantity of the first product across all displays.
2.  **Identify Model Type:** This is a Mixed Integer Linear Programming (MILP) problem (integer decision variables, linear objective and constraints).
3.  **Define Index Sets:** 
   - `S` = set of ShelfIDs (1,…,10) from *capacity.csv*.
   - `P` = set of Product indices (1,…,20) from *products.csv*, where the first index corresponds to the first product in the file (Smartphone).
4.  **Define Decision Variables:**
    - `x[i,j]` = number of units of product `j` placed on shelf `i`. Type: `GRB.INTEGER` (non‑negative).
5.  **Identify Parameters (from Schema):**
    - `Capacity[i]` from *capacity.csv*.
    - `Weight[j]` from *products.csv*.
    - `Value[j]` from *products.csv*.
6.  **Formulate Objective:**  
   \[
   \max \sum_{i \in S}\sum_{j \in P} \text{Value}[j] \cdot x[i,j]
   \]
7.  **Formulate Constraints:**
    - **Weight capacity on each shelf:**
      \[
      \sum_{j \in P} \text{Weight}[j] \cdot x[i,j] \;\le\; \text{Capacity}[i] \quad \forall i \in S
      \]
    - **Minimum quantity of the first product (j=1):**
      \[
      \sum_{i \in S} x[i,1] \;\ge\; 5
      \]
    - **Non‑negativity:**
      \[
      x[i,j] \;\ge\; 0 \quad \forall i \in S,\; j \in P
      \]
[Abstract Model Plan END]