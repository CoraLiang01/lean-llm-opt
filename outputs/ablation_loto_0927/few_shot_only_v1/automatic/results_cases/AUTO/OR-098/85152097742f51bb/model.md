[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work performed at their own home. The daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. Each worker contributes exactly 10 work days in total (including their own home). The work_days.csv file provides, for each homeowner (row), the number of days each worker (column) spent on that home.
2.  **Identify Model Type:** Based on the query, this is a system of linear equations (feasibility problem, not optimization), specifically a linear system for fair wage balancing with one fixed variable.
3.  **Define Index Sets:** The primary indices are Workers (W), with each worker both as a worker and as a homeowner (since each row is a homeowner, and each column is a worker).
4.  **Define Decision Variables:**
    -   `w[j]` = Daily wage rate for worker j (yuan per day). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Work days: `work_days[i][j]` = number of days worker j worked on homeowner i’s home (from work_days.csv, where i and j both run over all workers).
    -   Fixed wage: `w[Carpenter] = 60.00` (the first worker in the file).
    -   Each worker’s total work days: sum over i of work_days[i][j] = 10 for each worker j (given in the problem, not a constraint to enforce).
6.  **Formulate Objective:** There is no optimization objective; the goal is to find a feasible set of daily wages that satisfy the fairness constraints and the fixed wage for the first worker.
7.  **Formulate Constraints:**
    -   Constraint 1 (Fairness for each worker): For each worker k, the total income they earn from working on others’ homes equals the total they pay for work done at their own home:
        -   sum over i ≠ k of work_days[i][k] * w[k] = sum over j ≠ k of work_days[k][j] * w[j]
        -   Or, more generally, for each worker k:
            -   (sum over i of work_days[i][k]) * w[k] - work_days[k][k] * w[k] = (sum over j of work_days[k][j] * w[j]) - work_days[k][k] * w[k]
            -   Which simplifies to: (sum over i of work_days[i][k]) * w[k] = sum over j of work_days[k][j] * w[j]
    -   Constraint 2 (Fixed wage): w[Carpenter] = 60.00
    -   Constraint 3 (Optional, for interpretability): w[j] ≥ 0 for all workers j (wages should be non-negative).
[Abstract Model Plan END]