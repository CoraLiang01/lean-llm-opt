[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work done on their own home. The daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. Each worker contributes exactly 10 work days in total (across all projects, including their own).
2.  **Identify Model Type:** Based on the query, this is a system of linear equations (feasibility LP) with a normalization constraint (fixed wage for one worker).
3.  **Define Index Sets:** The primary indices are Workers (W), with each worker corresponding to a column (excluding the 'Owner' column). There are N = 150 workers (Carpenter, Electrician, Painter, Worker_004, ..., Worker_150).
4.  **Define Decision Variables:**
    -   `w[j]` = Daily wage of worker j (yuan per day). Type: GRB.CONTINUOUS, for all workers j in W.
5.  **Identify Parameters (from Schema):**
    -   Work days matrix: `work_days[i][j]` = Number of days worker j spent renovating owner i’s home. This is from the CSV, where row i corresponds to owner i, and column j to worker j.
    -   The mapping between owner and worker is by position: owner i’s home is row i, and worker j is column j.
    -   The daily wage of the first worker (Carpenter) is fixed: `w[Carpenter] = 60.00`.
    -   Each worker’s total work days: sum over all owners i of `work_days[i][j]` = 10 for each worker j.
6.  **Formulate Objective:** There is no explicit objective to optimize; the goal is to find feasible daily wages that satisfy the fairness constraints and the normalization (Carpenter’s wage fixed). (If needed, the model could minimize the sum of squared wage deviations or similar, but the query does not request this.)
7.  **Formulate Constraints:**
    -   **Fairness Constraint (for each worker k):** For every worker k in W:
        -   Total income from working on others’ homes = Total payment for work done at their own home.
        -   Income: sum over all owners i ≠ k of `work_days[i][k] * w[k]` (days worker k worked on others’ homes, times their wage).
        -   Payment: sum over all workers j ≠ k of `work_days[k][j] * w[j]` (days each worker j worked on worker k’s home, times wage of j).
        -   For each worker k:  
            `sum_{i ≠ k} work_days[i][k] * w[k] = sum_{j ≠ k} work_days[k][j] * w[j]`
        -   Or, equivalently, for all k:  
            `sum_{i} work_days[i][k] * w[k] - work_days[k][k] * w[k] = sum_{j} work_days[k][j] * w[j] - work_days[k][k] * w[k]`
            which simplifies to:  
            `sum_{i} work_days[i][k] * w[k] = sum_{j} work_days[k][j] * w[j]`
    -   **Normalization Constraint:**  
        -   `w[Carpenter] = 60.00`
    -   **(Implicit) Non-negativity Constraint:**  
        -   Optionally, require `w[j] ≥ 0` for all j (since negative wages are not meaningful).
[Abstract Model Plan END]