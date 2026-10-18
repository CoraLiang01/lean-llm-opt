[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 distinct machining machines to 12 distinct tasks such that each machine is used for exactly one task and each task is performed by exactly one machine, while minimizing the total machining cost given in `cost_12x12.csv`.

2.  **Identify Model Type:** This is a classic **assignment problem**, which is a special case of a **Mixed Integer Linear Programming (MILP)** model (specifically a 0‑1 linear assignment problem).

3.  **Define Index Sets:**  
    - `M` : set of machines, indexed by `i` (i = 1,…,12).  
    - `T` : set of tasks, indexed by `j` (j = 1,…,12).  
   The sets `M` and `T` are identical in size and correspond to the rows and columns of the cost matrix.

4.  **Define Decision Variables:**  
    - `x[i,j]` = 1 if machine `i` is assigned to task `j`; 0 otherwise.  
      Type: `GRB.BINARY`.  
    - (No auxiliary variables are needed for this pure assignment problem.)

5.  **Identify Parameters (from Schema):**  
    - **Cost matrix** `c[i,j]`: the machining cost for assigning machine `i` to task `j`.  
      These values are read from the columns `A`–`L` of `cost_12x12.csv`, with the row header `Machine` providing the machine index.  
    - No additional parameters (e.g., capacities or limits) are required because the only constraints are the one‑to‑one assignment restrictions.

6.  **Formulate Objective:**  
    Minimize the total machining cost:  
    \[
    \min \sum_{i \in M}\sum_{j \in T} c_{i,j}\,x_{i,j}
    \]

7.  **Formulate Constraints:**  
    - **Each machine is assigned to exactly one task:**  
      \[
      \forall i \in M:\quad \sum_{j \in T} x_{i,j} = 1
      \]
    - **Each task is performed by exactly one machine:**  
      \[
      \forall j \in T:\quad \sum_{i \in M} x_{i,j} = 1
      \]
    - **Binary nature of assignment variables:**  
      \[
      \forall i \in M,\;\forall j \in T:\quad x_{i,j} \in \{0,1\}
      \]

[Abstract Model Plan END]