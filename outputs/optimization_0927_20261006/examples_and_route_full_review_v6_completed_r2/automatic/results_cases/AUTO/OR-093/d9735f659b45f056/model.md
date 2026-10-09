[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 machines to 12 tasks, with each assignment incurring a specific cost from a provided cost matrix. The goal is to find a one-to-one assignment (each machine to exactly one task and vice versa) that minimizes the total cost.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) Assignment Problem (a special case of Integer Programming).
3.  **Define Index Sets:** The primary indices are Machines (M1–M12, from the 'Machine' column) and Tasks (A–L, from the column headers).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if machine i is assigned to task j; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost parameters are from the cost_12x12.csv file: `c_ij` is the value in row i (machine) and column j (task).
6.  **Formulate Objective:** Minimize the total assignment cost: sum over all machines and tasks of (c_ij * x[i,j]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Machine Assignment): For each machine i, sum over all tasks j of x[i,j] = 1 (each machine is assigned to exactly one task).
    -   Constraint 2 (Task Assignment): For each task j, sum over all machines i of x[i,j] = 1 (each task is assigned to exactly one machine).
[Abstract Model Plan END]