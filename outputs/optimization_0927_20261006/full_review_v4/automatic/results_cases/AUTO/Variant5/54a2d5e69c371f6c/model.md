[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each of a set of jobs (J1–J8) to one of several technician teams (M1–M4), minimizing total assignment cost, ensuring each job is assigned to exactly one team, and that no team's capacity is exceeded by the sum of the resources required for its assigned jobs. All assignment decisions are binary.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Generalized Assignment Problem (GAP).
3.  **Define Index Sets:** The primary indices are:
    - Teams (Machines): set of technician teams, indexed by i (from 'Machine' in all three CSVs).
    - Jobs: set of jobs, indexed by j (J1–J8, from columns in assignment_costs.csv and assignment_resources.csv).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to team i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost: `cost[i,j]` from assignment_costs.csv (row 'Machine' = i, column = j).
    -   Resource consumption: `resource[i,j]` from assignment_resources.csv (row 'Machine' = i, column = j).
    -   Team capacity: `capacity[i]` from machine_capacity.csv (row 'Machine' = i, column 'Capacity').
6.  **Formulate Objective:** Minimize total assignment cost: sum over all teams i and jobs j of (cost[i,j] * x[i,j]).
7.  **Formulate Constraints:**
    -   Assignment constraint: For each job j, sum over all teams i of x[i,j] = 1 (each job assigned to exactly one team).
    -   Capacity constraint: For each team i, sum over all jobs j of (resource[i,j] * x[i,j]) ≤ capacity[i] (team's total assigned workload does not exceed its capacity).
    -   Binary constraint: For all i, j, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]