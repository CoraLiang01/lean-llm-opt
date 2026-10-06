##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

##### Index Sets

- $M$: Set of managers, as listed in column "Unnamed: 0" of table_id "file_0_view_0"
- $P$: Set of projects, as listed in columns "P1", "P2", "P3" of table_id "file_0_view_0"

##### Parameters

- $c_{ij}$: Cost for manager $i$ to complete project $j$, from the entry in row with "Unnamed: 0" = $i$ and column $j$ in table_id "file_0_view_0"

##### Decision Variables

- $x_{ij}$: Binary variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise

##### Data Mapping

{
  "table_id": "file_0_view_0",
  "manager_id_column": "Unnamed: 0",
  "project_columns": ["P1", "P2", "P3"],
  "cost_parameter": "c_{ij} = \u2018file_0_view_0\u2019[manager_id_column = i, project_column = j]"
}