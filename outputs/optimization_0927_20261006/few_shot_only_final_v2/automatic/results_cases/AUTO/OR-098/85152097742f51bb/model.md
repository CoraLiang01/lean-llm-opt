[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work performed at their own home. The daily wage of the first worker (e.g., the Carpenter) is fixed at 60.00 yuan. Each worker contributes exactly 10 work days in total (including their own home). The work_days.csv file provides, for each homeowner (row), the number of days each worker (column) spent on that home.
2.  **Identify Model Type:** Based on the query, this is a system of linear equations (LP feasibility problem with a normalization constraint).
3.  **Define Index Sets:** The primary indices are Workers (W), where each worker is both a column and a row label in work_days.csv (excluding the 'Owner' column).
4.  **Define Decision Variables:**
    -   `w[j]` = Daily wage of worker j. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   `D[i,j]` = Number of days worker j worked on homeowner i’s home. From work_days.csv, where i and j are both in the set of workers (excluding the 'Owner' column for j, and using the 'Owner' column for i).
    -   The set of workers is the list of columns (excluding 'Owner'), and the set of homeowners is the list of 'Owner' values (which matches the worker list).
6.  **Formulate Objective:** There is no optimization objective; the goal is to find feasible daily wages satisfying all constraints, with the additional normalization that the first worker’s wage is fixed at 60.00 yuan.
7.  **Formulate Constraints:**
    -   **Mutual Payment Balance (for each worker i):** The total income worker i earns from working on others’ homes equals the total payment they make for work done at their own home. For each worker i:
        -   sum over all j ≠ i of D[j,i] * w[i] = sum over all j ≠ i of D[i,j] * w[j]
        -   (Or, equivalently: sum over all j of D[j,i] * w[i] = sum over all j of D[i,j] * w[j], since D[i,i] * w[i] appears on both sides and cancels.)
    -   **Wage Normalization:** The daily wage of the first worker (the first column in work_days.csv, e.g., 'Carpenter') is fixed: w[first_worker] = 60.00.
    -   **(Implicit) Non-negativity:** Optionally, require w[j] ≥ 0 for all j (if negative wages are not meaningful).
[Abstract Model Plan END]