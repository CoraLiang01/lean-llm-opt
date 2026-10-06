[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign 12 machines to 12 tasks, with each machine assigned to exactly one task and each task to exactly one machine, in order to minimize the total machining cost. The cost of assigning machine i to task j is given in cost_12x12.csv.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) Assignment Problem (specifically, a one-to-one minimum-cost assignment).
3.  **Define Index Sets:** The primary indices are:
    - Machines (set of 12, e.g., M1, M2, ..., M12 from the 'Machine' column)
    - Tasks (set of 12, e.g., A, B, ..., L from the column headers)
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if machine i is assigned to task j, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost coefficients come from the intersection of the 'Machine' row and the task columns ('A' through 'L') in cost_12x12.csv; that is, c_ij = schema['task'][i] for each machine-task pair.
    -   No additional parameters are needed; all constraints are structural (assignment).
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., minimize sum over all machines i and tasks j of (c_ij * x[i,j]), where c_ij is the cost from the CSV.
7.  **Formulate Constraints:**
    -   Constraint 1 (Machine Assignment): For each machine i, sum over all tasks j of x[i,j] = 1 (each machine is assigned to exactly one task).
    -   Constraint 2 (Task Assignment): For each task j, sum over all machines i of x[i,j] = 1 (each task is assigned to exactly one machine).
    -   Constraint 3 (Binary): x[i,j] ∈ {0,1} for all machine-task pairs (enforces one-to-one assignment).
[Abstract Model Plan END]