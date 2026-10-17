[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 machines to 12 tasks, with each machine assigned to exactly one task and each task to exactly one machine, in order to minimize the total machining cost. The assignment costs are given in a 12x12 cost matrix from the CSV file.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) Assignment Problem (a special case of Integer Programming).
3.  **Define Index Sets:** The primary indices are:
    - Machines (set M): {M1, M2, ..., M12} (from the 'Machine' column)
    - Tasks (set T): {A, B, ..., L} (from the column headers, excluding 'Machine')
4.  **Define Decision Variables:**
    -   `x[m, t]` = 1 if machine m is assigned to task t; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost `c[m, t]` comes from the value in row m and column t of cost_12x12.csv.
6.  **Formulate Objective:** Minimize the total assignment cost: sum over all machines and tasks of (c[m, t] * x[m, t]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Machine Assignment): For each machine m in M, sum over all tasks t in T of x[m, t] = 1 (each machine assigned to exactly one task).
    -   Constraint 2 (Task Assignment): For each task t in T, sum over all machines m in M of x[m, t] = 1 (each task assigned to exactly one machine).
[Abstract Model Plan END]