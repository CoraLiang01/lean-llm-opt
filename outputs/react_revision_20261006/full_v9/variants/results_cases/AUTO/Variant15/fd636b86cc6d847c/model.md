##### Objective Function:

$\quad \min \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints (Each job assigned to exactly one workstation):

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

###### 2. Capacity Constraints (Workstation capacity not exceeded):

$\sum_{j \in J} a_{ij} \, x_{ij} \leq b_i \quad \forall i \in I$

###### 3. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in I, \; j \in J$

---

##### Data Mapping

- $I$: Set of workstations, from file_0_view_0["Workstation"], file_1_view_0["Workstation"], file_2_view_0["Workstation"]
- $J$: Set of jobs, from file_1_view_0 columns ["J1", "J2", ..., "J9"], file_2_view_0 columns ["J1", ..., "J9"]
- $c_{ij}$: Assignment cost of assigning job $j$ to workstation $i$, from file_1_view_0, row "Workstation" $i$, column $j$
- $a_{ij}$: Resource consumed by assigning job $j$ to workstation $i$, from file_2_view_0, row "Workstation" $i$, column $j$
- $b_i$: Capacity of workstation $i$, from file_0_view_0, row "Workstation" $i$, column "Capacity"
- $x_{ij}$: Binary variable, 1 if job $j$ is assigned to workstation $i$, 0 otherwise

All index sets, parameters, and coefficients are defined exactly as in the retrieved CSV data.