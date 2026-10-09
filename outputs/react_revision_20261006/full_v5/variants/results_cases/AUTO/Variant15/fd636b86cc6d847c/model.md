[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each custom job to exactly one workstation in a way that minimizes total assignment cost, while ensuring that the total resource consumption at each workstation does not exceed its capacity. The assignment cost and resource consumption for each workstation-job pair are given, as are the workstation capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a minimum-cost Generalized Assignment Problem (GAP).
3.  **Define Index Sets:** The primary indices are:
    - Workstations (from 'Workstation' column in all three CSVs): W1, W2, W3, W4
    - Jobs (from columns J1–J9 in assignment_costs.csv and assignment_resources.csv): J1, J2, ..., J9
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to workstation i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Assignment cost for each workstation-job pair: from assignment_costs.csv, columns J1–J9 for each 'Workstation'.
    -   Resource consumption for each workstation-job pair: from assignment_resources.csv, columns J1–J9 for each 'Workstation'.
    -   Workstation capacity: from workstation_capacity.csv, 'Capacity' column for each 'Workstation'.
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., sum over all workstations and jobs of (assignment cost for i-j) × x[i,j].
7.  **Formulate Constraints:**
    -   Constraint 1 (Exactly-One Assignment per Job): For each job j, the sum over all workstations i of x[i,j] = 1 (i.e., each job is assigned to exactly one workstation).
    -   Constraint 2 (Workstation Capacity): For each workstation i, the sum over all jobs j of (resource consumption for i-j) × x[i,j] ≤ capacity of workstation i.
    -   Constraint 3 (Binary Variables): For all i, j, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]