##### Objective Function:

$\quad \min \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}$

where $x_{ij}$ is a binary variable equal to 1 if job $j$ is assigned to workstation $i$, and 0 otherwise. $c_{ij}$ is the assignment cost for assigning job $j$ to workstation $i$.

##### Constraints:

###### 1. Each job is assigned to exactly one workstation:

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

###### 2. Workstation capacity constraints:

$\sum_{j \in J} a_{ij} \, x_{ij} \leq b_i \quad \forall i \in I$

where $a_{ij}$ is the resource consumed by assigning job $j$ to workstation $i$, and $b_i$ is the capacity of workstation $i$.

###### 3. Binary assignment variables:

$x_{ij} \in \{0,1\} \quad \forall i \in I, \forall j \in J$

##### Retrieved Information

{
  "workstations": [
    "W1",
    "W2",
    "W3",
    "W4"
  ],
  "jobs": [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
    "J7",
    "J8",
    "J9"
  ],
  "capacity": {
    "table_id": "file_0_view_0",
    "column_names": ["Workstation", "Capacity"]
  },
  "assignment_costs": {
    "table_id": "file_1_view_0",
    "row_id": "Workstation",
    "column_names": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
  },
  "assignment_resources": {
    "table_id": "file_2_view_0",
    "row_id": "Workstation",
    "column_names": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
  }
}