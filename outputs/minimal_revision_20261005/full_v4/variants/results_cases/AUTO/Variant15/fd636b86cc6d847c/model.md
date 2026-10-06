##### Objective Function:

$\quad \min \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}$

##### Constraints:

###### 1. Assignment Constraints (Each job assigned to exactly one workstation):

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

###### 2. Capacity Constraints (Workstation capacity not exceeded):

$\sum_{j \in J} a_{ij} \, x_{ij} \leq b_i \quad \forall i \in I$

###### 3. Binary Assignment Variables:

$x_{ij} \in \{0,1\} \quad \forall i \in I, \; j \in J$

---

##### Data Mapping

- $I$: Set of workstations, from `file_0_view_0.Workstation`
- $J$: Set of jobs, from `file_1_view_0` and `file_2_view_0` columns $\{J1, J2, J3, J4, J5, J6, J7, J8, J9\}$
- $c_{ij}$: Assignment cost of job $j$ to workstation $i$, from `file_1_view_0` with row index $i$ (`Workstation`) and column $j$
- $a_{ij}$: Resource consumed by assigning job $j$ to workstation $i$, from `file_2_view_0` with row index $i$ (`Workstation`) and column $j$
- $b_i$: Capacity of workstation $i$, from `file_0_view_0` column `Capacity` with row index $i$ (`Workstation`)

---

##### Variable Definition

- $x_{ij} = \begin{cases} 1 & \text{if job } j \text{ is assigned to workstation } i \\ 0 & \text{otherwise} \end{cases}$

---

##### Source Data Mapping

```json
{
  "workstations": {
    "table_id": "file_0_view_0",
    "column": "Workstation"
  },
  "jobs": {
    "table_id": "file_1_view_0",
    "columns": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
  },
  "assignment_costs": {
    "table_id": "file_1_view_0",
    "row_id": "Workstation",
    "columns": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
  },
  "assignment_resources": {
    "table_id": "file_2_view_0",
    "row_id": "Workstation",
    "columns": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
  },
  "workstation_capacity": {
    "table_id": "file_0_view_0",
    "row_id": "Workstation",
    "column": "Capacity"
  }
}
```