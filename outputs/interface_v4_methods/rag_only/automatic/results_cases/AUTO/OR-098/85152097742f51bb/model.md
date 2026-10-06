[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work done on their own home. The daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. The work_days.csv file provides, for each homeowner (row), the number of days each worker (column) spent on that home. Each worker contributes exactly 10 work days in total.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) feasibility problem (system of linear equations with one fixed variable).
3.  **Define Index Sets:** The primary indices are:
    -   Workers: All columns except 'Owner' (i.e., ['Carpenter', 'Electrician', 'Painter', ..., 'Worker_150']), denoted as set W.
    -   Homeowners: All rows (each row corresponds to a homeowner), denoted as set H. Each homeowner is also a worker.
4.  **Define Decision Variables:**
    -   `w[j]` = Daily wage of worker j (for all j in W). Type: GRB.CONTINUOUS.
        -   Note: `w[Carpenter]` is fixed at 60.00 yuan; all other `w[j]` are variables to be determined.
5.  **Identify Parameters (from Schema):**
    -   `days[h][j]`: Number of days worker j worked on homeowner h’s home. From the CSV, this is the value at row h, column j.
    -   The mapping between homeowners and workers is one-to-one: homeowner h is also worker h (i.e., row h corresponds to worker j=h).
6.  **Formulate Objective:** There is no optimization objective; the goal is to find feasible daily wages that satisfy the fairness constraints and the fixed wage for the Carpenter.
7.  **Formulate Constraints:**
    -   Constraint 1 (Fairness for each participant): For each worker/homeowner i in W/H,
        -   The total income worker i earns from working on others’ homes equals the total amount they pay for work done on their own home:
        -   sum over h ≠ i of [days[h][i] * w[i]] = sum over j ≠ i of [days[i][j] * w[j]]
        -   Or, equivalently, for each i:
            -   (sum over h of days[h][i]) * w[i] - days[i][i] * w[i] = (sum over j of days[i][j] * w[j]) - days[i][i] * w[i]
            -   Which simplifies to: (sum over h of days[h][i]) * w[i] = sum over j of days[i][j] * w[j]
    -   Constraint 2 (Fixed wage): w[Carpenter] = 60.00
    -   Constraint 3 (Optional, if needed for numerical stability): w[j] ≥ 0 for all j (wages cannot be negative).
    -   (Note: The constraint that each worker contributes exactly 10 work days is already enforced by the data and does not need to be modeled.)
[Abstract Model Plan END]