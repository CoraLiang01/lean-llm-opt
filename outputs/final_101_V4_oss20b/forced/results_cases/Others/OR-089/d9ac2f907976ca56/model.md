[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to open a subset of ten candidate service centers to serve fifteen customers, minimizing total fixed opening costs plus customer–centre service costs, with each customer assigned to exactly one opened centre and each centre serving at most four customers.
2.  **Identify Model Type:** This is a Fixed‑Charge Facility Location problem, a Mixed Integer Linear Programming (MIP) model with binary opening and assignment variables.
3.  **Define Index Sets:** 
    - `J` = {SC1, SC2, …, SC10} – the set of candidate service centers.  
    - `I` = {C1, C2, …, C15} – the set of customers.
4.  **Define Decision Variables:**
    - `y[j]` = 1 if service center `j` is opened, 0 otherwise. Type: `GRB.BINARY`.  
    - `x[i,j]` = 1 if customer `i` is assigned to opened service center `j`, 0 otherwise. Type: `GRB.BINARY`.
5.  **Identify Parameters (from Schema):**
    - Fixed opening costs: `f[j]` from column “Fixed Opening Cost” in `service_centers_fixed_costs.csv`.  
    - Service costs: `c[i,j]` from column `j` in `expanded_customer_service_costs.csv` for each customer `i`.  
    - Capacity limit: each centre can serve at most 4 customers (constant `U = 4`).
6.  **Formulate Objective:** Minimize  
   \[
   \sum_{j\in J} f[j]\,y[j] \;+\; \sum_{i\in I}\sum_{j\in J} c[i,j]\,x[i,j].
   \]
7.  **Formulate Constraints:**
    - **Customer assignment:**  
      \[
      \sum_{j\in J} x[i,j] = 1 \quad \forall i\in I.
      \]
    - **Linking (assignment only to opened centres):**  
      \[
      x[i,j] \le y[j] \quad \forall i\in I,\; j\in J.
      \]
    - **Capacity of each centre:**  
      \[
      \sum_{i\in I} x[i,j] \le 4 \quad \forall j\in J.
      \]
[Abstract Model Plan END]