##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "row_index_set": "M = {MA, MB, MC, MD, ME, MF} (from column 'Unnamed: 0')",
    "column_index_set": "P = {P1, P2, P3, P4, P5, P6} (from columns 'P1', 'P2', 'P3', 'P4', 'P5', 'P6')",
    "mapping": "c_{ij} = value at row i (manager), column j (project) in 'manager_project_costs.csv', table_id 'file_0_view_0'"
  },
  "managers": [
    "MA",
    "MB",
    "MC",
    "MD",
    "ME",
    "MF"
  ],
  "projects": [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6"
  ]
}