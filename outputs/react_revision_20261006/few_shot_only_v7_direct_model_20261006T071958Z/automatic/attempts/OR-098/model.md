[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work performed at their own home. The daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. Each worker contributes exactly 10 work days in total (including their own home). The work_days.csv file provides, for each homeowner (row), the number of days each worker (column) spent on that home.
2.  **Identify Model Type:** Based on the query, this is a system of linear equations (feasibility LP) with a normalization constraint (fixed wage for one worker).
3.  **Define Index Sets:** The primary indices are Workers (W), where each worker is both a column and a row (as Owner) in the CSV. Let W = {Carpenter, Electrician, Painter, Worker_004, ..., Worker_150}.
4.  **Define Decision Variables:**
    -   `w[j]` = Daily wage of worker j (for all j in W). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   `days[i][j]` = Number of days worker j spent working on homeowner i’s home. From work_days.csv, where i = Owner, j = column name.
    -   The fixed wage: w[Carpenter] = 60.00 (Carpenter is the first worker listed).
    -   Each worker’s total work days: sum over i of days[i][j] = 10 for all j (given in the query, not a constraint to enforce).
6.  **Formulate Objective:** There is no optimization objective; the goal is to find a feasible set of daily wages that satisfy the fairness constraints and the normalization (Carpenter’s wage fixed). (If needed, the model could minimize the sum of squared wage deviations or similar, but the query does not request this.)
7.  **Formulate Constraints:**
    -   **Fairness Constraint (for each worker k in W):** The total income worker k earns from working on others’ homes equals the total amount they pay for work performed at their own home. For each worker k:
        -   sum over i ≠ k of days[i][k] * w[k] = sum over j ≠ k of days[k][j] * w[j]
        -   (Alternatively, sum over all i of days[i][k] * w[k] - days[k][k] * w[k] = sum over all j of days[k][j] * w[j] - days[k][k] * w[k])
        -   This ensures that for each worker, their income from others equals their payment to others.
    -   **Normalization Constraint:** w[Carpenter] = 60.00.
    -   **Non-negativity Constraint:** w[j] ≥ 0 for all j in W (wages cannot be negative).
[Abstract Model Plan END]