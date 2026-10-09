[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each of a set of jobs (J1–J8) to one of several technician teams (M1–M4), minimizing total assignment cost, ensuring each job is assigned to exactly one team, and that no team's capacity is exceeded by the sum of the resources required for its assigned jobs. All assignment decisions are binary.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Generalized Assignment Problem (GAP).
3.  **Define Index Sets:** The primary indices are Teams (Machines: M1, M2, M3, M4) and Jobs (J1, J2, ..., J8).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to team i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost: from assignment_costs.csv, field [Machine][J1–J8] (cost of assigning job j to team i).
    -   Resource consumption: from assignment_resources.csv, field [Machine][J1–J8] (capacity consumed on team i by job j).
    -   Team capacity: from machine_capacity.csv, field [Machine][Capacity] (total available capacity for team i).
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., sum over all teams and jobs of (assignment cost for team i and job j) × x[i,j].
7.  **Formulate Constraints:**
    -   Assignment Constraint: For each job j, sum over all teams i of x[i,j] = 1 (each job assigned to exactly one team).
    -   Capacity Constraint: For each team i, sum over all jobs j of (resource consumption for team i and job j) × x[i,j] ≤ team i's capacity.
    -   Binary Constraint: For all i, j, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]