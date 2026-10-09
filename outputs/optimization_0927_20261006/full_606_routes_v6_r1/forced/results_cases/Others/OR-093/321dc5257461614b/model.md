[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 machines to 12 tasks, with each machine assigned to exactly one task and each task to exactly one machine, in order to minimize the total machining cost. The cost of assigning machine i to task j is given in the cost_12x12.csv file.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) Assignment Problem (a special case of Integer Programming).
3.  **Define Index Sets:** The primary indices are Machines (M1–M12, from the 'Machine' column) and Tasks (A–L, from the column headers).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if machine i is assigned to task j; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost parameters: c_ij, from the intersection of row 'Machine' i and column j (A–L) in cost_12x12.csv.
6.  **Formulate Objective:** Minimize the total assignment cost: sum over all machines i and tasks j of c_ij * x[i,j].
7.  **Formulate Constraints:**
    -   Constraint 1 (Machine Assignment): For each machine i, sum over all tasks j of x[i,j] = 1 (each machine is assigned to exactly one task).
    -   Constraint 2 (Task Assignment): For each task j, sum over all machines i of x[i,j] = 1 (each task is assigned to exactly one machine).
[Abstract Model Plan END]