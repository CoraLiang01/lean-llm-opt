[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 machines to 12 tasks, with each machine assigned to exactly one task and each task assigned to exactly one machine, in order to minimize the total machining cost. The cost of assigning machine i to task j is given in the cost_12x12.csv file.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) Assignment Problem (specifically, a one-to-one minimum-cost assignment).
3.  **Define Index Sets:** The primary indices are:
    - Machines (set of 12, e.g., M1, M2, ..., M12)
    - Tasks (set of 12, e.g., A, B, ..., L)
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if machine i is assigned to task j; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost coefficients come from the CSV columns: for each machine (row, 'Machine'), the cost to assign to each task (columns 'A' through 'L').
    -   There are no additional parameters needed; all costs are directly from the CSV.
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., minimize the sum over all machines and tasks of (cost_ij * x[i,j]), where cost_ij is the value from the CSV for machine i and task j.
7.  **Formulate Constraints:**
    -   Constraint 1 (Machine Assignment): For each machine i, sum over all tasks j of x[i,j] = 1 (each machine is assigned to exactly one task).
    -   Constraint 2 (Task Assignment): For each task j, sum over all machines i of x[i,j] = 1 (each task is assigned to exactly one machine).
    -   Constraint 3 (Binary): x[i,j] ∈ {0,1} for all machines i and tasks j.
[Abstract Model Plan END]