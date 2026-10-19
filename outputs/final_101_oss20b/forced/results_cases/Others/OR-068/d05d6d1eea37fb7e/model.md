[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each of 6 managers to exactly one of 6 construction projects at minimum total cost, using the cost matrix provided in *manager_project_costs.csv*.
2.  **Identify Model Type:** This is a **Mixed Integer Programming (MIP)** problem, specifically a classic assignment (one‑to‑one matching) problem.
3.  **Define Index Sets:**  
    - `M` = {MA, MB, MC, MD, ME, MF}  (set of managers, extracted from the first column “Unnamed: 0”).  
    - `P` = {P1, P2, P3, P4, P5, P6}  (set of projects, extracted from the column headers).
4.  **Define Decision Variables:**  
    - `x[m,p]` = 1 if manager *m* is assigned to project *p*, 0 otherwise.  
      Type: `GRB.BINARY`.  
5.  **Identify Parameters (from Schema):**  
    - `c[m,p]` = cost of assigning manager *m* to project *p*, taken directly from the corresponding cell in the CSV.  
6.  **Formulate Objective:**  
    - **Minimize** the total assignment cost:  
      \[
      \min \sum_{m \in M}\sum_{p \in P} c[m,p] \cdot x[m,p]
      \]
7.  **Formulate Constraints:**  
    - **Each manager is assigned to exactly one project:**  
      \[
      \forall m \in M:\quad \sum_{p \in P} x[m,p] = 1
      \]
    - **Each project receives exactly one manager:**  
      \[
      \forall p \in P:\quad \sum_{m \in M} x[m,p] = 1
      \]
    - **Binary nature of variables:**  
      \[
      \forall m \in M,\forall p \in P:\quad x[m,p] \in \{0,1\}
      \]
[Abstract Model Plan END]