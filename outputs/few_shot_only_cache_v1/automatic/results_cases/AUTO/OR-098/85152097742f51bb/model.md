[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a fair daily wage for each worker in a mutual-aid renovation group, such that for every participant, the total income they earn from working on others’ homes equals the total amount they pay for work done at their own home. The daily wage of the first worker (Carpenter) is fixed at 60.00 yuan. Each worker contributes exactly 10 work days in total across all projects (including their own).
2.  **Identify Model Type:** Based on the query, this is a system of linear equations (feasibility LP) with a normalization constraint (fixed wage for one worker).
3.  **Define Index Sets:** The primary indices are Workers (columns "Carpenter", "Electrician", "Painter", ..., "Worker_150") and Owners (rows, each corresponding to a worker’s home).
4.  **Define Decision Variables:**
    -   `w[j]` = daily wage of worker j (for all workers, including Carpenter). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Work days matrix: `D[i,j]` = number of days worker j worked on owner i’s home (from work_days.csv, columns as workers, rows as owners).
    -   Fixed wage: `w[Carpenter] = 60.00` (Carpenter is the first worker listed).
    -   Each worker’s total work days: sum over i of D[i,j] = 10 for all j (given in the query).
6.  **Formulate Objective:** There is no optimization objective; the goal is to find feasible daily wages satisfying all constraints (feasibility problem).
7.  **Formulate Constraints:**
    -   **Mutual Payment Balance (for each worker k):**  
        For every worker k,  
        sum over j≠k of D[k,j] * w[j] = sum over i≠k of D[i,k] * w[k]  
        (i.e., total payment worker k makes for others working on their home equals total income worker k earns from working on others’ homes).
    -   **Fixed Wage Constraint:**  
        w[Carpenter] = 60.00
    -   **Total Work Days Constraint:**  
        For all workers j, sum over i of D[i,j] = 10 (already satisfied by the data; no need to enforce in the model).
    -   **Non-negativity:**  
        w[j] ≥ 0 for all workers j (wages cannot be negative).
[Abstract Model Plan END]