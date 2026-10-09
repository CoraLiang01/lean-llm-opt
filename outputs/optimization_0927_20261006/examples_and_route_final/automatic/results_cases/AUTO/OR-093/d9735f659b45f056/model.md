[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 machines to 12 tasks in a one-to-one manner, minimizing the total machining cost, where the cost of assigning machine i to task j is given in the cost_12x12.csv file.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) Assignment Problem (a special case of Integer Programming).
3.  **Define Index Sets:** The primary indices are Machines (i ∈ {M1, ..., M12}) and Tasks (j ∈ {A, ..., L}).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if machine i is assigned to task j; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost parameters: c_ij, from the intersection of row 'Machine' (M1–M12) and columns 'A'–'L' in cost_12x12.csv.
6.  **Formulate Objective:** Minimize the total assignment cost: sum over all machines and tasks of c_ij * x[i,j].
7.  **Formulate Constraints:**
    -   Assignment Constraint (Machines): For each machine i, sum over all tasks j of x[i,j] = 1 (each machine assigned to exactly one task).
    -   Assignment Constraint (Tasks): For each task j, sum over all machines i of x[i,j] = 1 (each task assigned to exactly one machine).
[Abstract Model Plan END]