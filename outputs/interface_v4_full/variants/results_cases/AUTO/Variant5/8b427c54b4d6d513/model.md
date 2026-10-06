[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each of a set of jobs (J1–J8) to technician teams (M1–M4) such that every job is assigned to exactly one team, the total capacity consumed by each team does not exceed its available capacity, and the total assignment cost is minimized. Each assignment consumes a specific amount of team capacity and incurs a specific cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Generalized Assignment Problem (GAP).
3.  **Define Index Sets:** The primary indices are:
    - Teams (Machines): i ∈ {M1, M2, M3, M4}
    - Jobs: j ∈ {J1, J2, J3, J4, J5, J6, J7, J8}
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to team i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost: from assignment_costs.csv, columns 'J1'–'J8' for each 'Machine' (team).
    -   Capacity consumed per assignment: from assignment_resources.csv, columns 'J1'–'J8' for each 'Machine'.
    -   Team capacity limits: from machine_capacity.csv, column 'Capacity' for each 'Machine'.
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., sum over all teams and jobs of (assignment cost for team i and job j) × x[i,j].
7.  **Formulate Constraints:**
    -   Constraint 1 (Job Assignment): For each job j, sum over all teams i of x[i,j] = 1 (each job is assigned to exactly one team).
    -   Constraint 2 (Team Capacity): For each team i, sum over all jobs j of (capacity consumed by assigning job j to team i) × x[i,j] ≤ team i's available capacity.
    -   Constraint 3 (Binary Variables): For all i, j, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]