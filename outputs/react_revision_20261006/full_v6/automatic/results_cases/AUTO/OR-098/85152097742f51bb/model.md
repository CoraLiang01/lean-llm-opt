[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work done on their own home. The daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. Each worker contributes exactly 10 work days in total (across all projects, including their own).
2.  **Identify Model Type:** Based on the query, this is a system of linear equations (feasibility problem), specifically a wage balancing problem (can be formulated as an LP with equality constraints and a fixed variable).
3.  **Define Index Sets:** The primary indices are Workers (columns excluding 'Owner'), and Homeowners (rows, each corresponding to a worker’s home).
4.  **Define Decision Variables:**
    -   `w[j]` = Daily wage rate for worker j (for all workers, with `w[Carpenter]` fixed at 60.00). Type: CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Work days matrix: `work_days[i][j]` = number of days worker j worked on homeowner i’s home (from the CSV, where i indexes rows and j indexes worker columns).
    -   Worker list: All columns except 'Owner' are workers; each row’s 'Owner' identifies the homeowner (who is also a worker).
    -   Each worker’s own home: For worker j, their own home is the row where 'Owner' == worker j.
6.  **Formulate Objective:** There is no explicit optimization objective; the goal is to find feasible daily wages that satisfy all balancing constraints and the fixed wage for the first worker. (If needed, the objective can be to minimize the sum of squared wage deviations, but the query does not request this.)
7.  **Formulate Constraints:**
    -   Constraint 1 (Income-Expense Balance for Each Worker): For each worker j,  
        -   Total income from working on others’ homes:  
            `sum over i ≠ j of work_days[i][j] * w[j]`
        -   Total expense for work done at their own home:  
            `sum over k ≠ j of work_days[j][k] * w[k]`
        -   Balance constraint:  
            `sum_{i ≠ j} work_days[i][j] * w[j] = sum_{k ≠ j} work_days[j][k] * w[k]`  
            (Or, equivalently, for all j:  
            `w[j] * sum_{i ≠ j} work_days[i][j] - sum_{k ≠ j} work_days[j][k] * w[k] = 0`)
    -   Constraint 2 (Fixed Wage):  
        -   `w[Carpenter] = 60.00`
    -   Constraint 3 (Optional, if needed):  
        -   Non-negativity: `w[j] ≥ 0` for all j (if negative wages are not meaningful).
    -   Constraint 4 (Total Work Days):  
        -   Each worker’s total work days (sum over all homeowners) is exactly 10:  
            `sum_{i} work_days[i][j] = 10` for all j (this is given by the problem and can be used for validation, but not needed as a constraint if already satisfied in the data).
[Abstract Model Plan END]