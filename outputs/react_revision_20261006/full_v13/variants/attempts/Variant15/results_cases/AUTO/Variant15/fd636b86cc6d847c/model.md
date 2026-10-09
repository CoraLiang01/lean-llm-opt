##### Objective Function:

$\quad \min \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints (each job assigned to exactly one workstation):

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

###### 2. Capacity Constraints (workstation capacity not exceeded):

$\sum_{j \in J} a_{ij} \, x_{ij} \leq b_i \quad \forall i \in I$

###### 3. Binary Assignment Variables:

$x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J$

---

##### Data Mapping

- $I$: Set of workstations, from column "Workstation" in file_0_view_0, file_1_view_0, and file_2_view_0.
- $J$: Set of jobs, from columns "J1"–"J9" in file_1_view_0 and file_2_view_0.
- $b_i$: Capacity of workstation $i$, from column "Capacity" in file_0_view_0.
- $c_{ij}$: Assignment cost for workstation $i$ and job $j$, from file_1_view_0, row with "Workstation" = $i$, column $j$.
- $a_{ij}$: Resource consumed if workstation $i$ is assigned job $j$, from file_2_view_0, row with "Workstation" = $i$, column $j$.
- $x_{ij}$: Binary variable, 1 if job $j$ is assigned to workstation $i$, 0 otherwise.