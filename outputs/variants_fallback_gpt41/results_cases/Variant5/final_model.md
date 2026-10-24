[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each of a set of jobs (J1–J8) to one of several technician teams (M1–M4), minimizing total assignment cost. Each assignment consumes some of the team's capacity, and no team's total assigned workload may exceed its capacity. Each job must be assigned to exactly one team, and assignments are binary (either assigned or not).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Generalized Assignment Problem (GAP).
3.  **Define Index Sets:** The primary indices are:
    - Teams (i ∈ Machines, from 'Machine' in all three CSVs)
    - Jobs (j ∈ Jobs, J1–J8, from columns in assignment_costs.csv and assignment_resources.csv)
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to team i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost: from assignment_costs.csv, field [Machine][J1–J8] (cost of assigning job j to team i).
    -   Capacity consumed: from assignment_resources.csv, field [Machine][J1–J8] (capacity used by assigning job j to team i).
    -   Team capacity: from machine_capacity.csv, field [Machine][Capacity] (total available capacity for each team).
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., sum over all teams and jobs of (assignment cost for team i and job j) × x[i,j].
7.  **Formulate Constraints:**
    -   Constraint 1 (Job Assignment): For each job j, sum over all teams i of x[i,j] = 1 (each job assigned to exactly one team).
    -   Constraint 2 (Team Capacity): For each team i, sum over all jobs j of (capacity consumed by assigning job j to team i) × x[i,j] ≤ team i's capacity.
    -   Constraint 3 (Binary Variables): For all i, j, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]