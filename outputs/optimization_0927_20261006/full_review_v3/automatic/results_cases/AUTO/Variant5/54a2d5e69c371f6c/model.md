[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each of a set of jobs (J1–J8) to one of several technician teams (M1–M4), minimizing total assignment cost, ensuring each job is assigned to exactly one team, and that no team's capacity is exceeded by the sum of the resources required for its assigned jobs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) assignment problem (Generalized Assignment Problem).
3.  **Define Index Sets:** The primary indices are Teams (Machines: M1–M4) and Jobs (J1–J8).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to team i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost for each team-job pair: from assignment_costs.csv, columns 'J1'–'J8' for each 'Machine'.
    -   Resource (capacity) consumed by each assignment: from assignment_resources.csv, columns 'J1'–'J8' for each 'Machine'.
    -   Team capacity limits: from machine_capacity.csv, column 'Capacity' for each 'Machine'.
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., sum over all teams and jobs of (assignment cost for team i and job j) × x[i,j].
7.  **Formulate Constraints:**
    -   Assignment Constraint: For each job j, sum over all teams i of x[i,j] = 1 (each job assigned to exactly one team).
    -   Capacity Constraint: For each team i, sum over all jobs j of (resource consumed by assigning job j to team i) × x[i,j] ≤ team i's capacity.
    -   Binary Constraint: For all i, j, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]