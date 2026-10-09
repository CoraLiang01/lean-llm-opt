[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to assign each custom job to exactly one workstation, minimizing total assignment cost, while ensuring that the total resource consumption at each workstation does not exceed its capacity. Assignment costs and resource consumptions are specific to each workstation-job pair.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Generalized Assignment Problem (GAP).
3.  **Define Index Sets:** The primary indices are Workstations (from 'Workstation' in all three files) and Jobs (from 'J1'–'J9' columns in assignment_costs.csv and assignment_resources.csv).
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if job j is assigned to workstation i; 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Assignment costs for each workstation-job pair, from assignment_costs.csv (fields: 'Workstation', 'J1'–'J9').
    -   Constraint coefficients: Resource consumption for each workstation-job pair, from assignment_resources.csv (fields: 'Workstation', 'J1'–'J9').
    -   Constraint RHS: Workstation capacities, from workstation_capacity.csv (fields: 'Workstation', 'Capacity').
6.  **Formulate Objective:** Minimize the total assignment cost, i.e., sum over all workstations and jobs of (assignment cost for i-j) × x[i,j].
7.  **Formulate Constraints:**
    -   Assignment Constraint: For each job, the sum over all workstations of x[i,j] = 1 (each job assigned to exactly one workstation).
    -   Capacity Constraint: For each workstation, the sum over all jobs of (resource consumption for i-j) × x[i,j] ≤ capacity of workstation i.
    -   Binary Restriction: For all workstation-job pairs, x[i,j] ∈ {0,1}.
[Abstract Model Plan END]