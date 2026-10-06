##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Index Sets

- $M$: Set of managers, as listed in column "Unnamed: 0" of table_id "file_0_view_0"
- $P$: Set of projects, as listed in columns ["P1", "P2", "P3", "P4", "P5", "P6"] of table_id "file_0_view_0"

##### Parameters

- $c_{ij}$: Cost of assigning manager $i$ to project $j$, from table_id "file_0_view_0", row with "Unnamed: 0" = $i$, column $j$

##### Decision Variables

- $x_{ij}$: $=1$ if manager $i$ is assigned to project $j$, $0$ otherwise

##### Data Mapping

{
  "table_id": "file_0_view_0",
  "manager_id_column": "Unnamed: 0",
  "project_columns": ["P1", "P2", "P3", "P4", "P5", "P6"],
  "cost_parameter": "c_{ij} = \u201cfile_0_view_0\u201d[manager_id_column=i, project_column=j]"
}