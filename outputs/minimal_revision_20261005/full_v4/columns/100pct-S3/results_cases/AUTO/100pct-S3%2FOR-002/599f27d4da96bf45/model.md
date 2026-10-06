##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Index Sets

- $M$: Set of managers, as listed in column "Unnamed: 0" of table_id "file_0_view_0"
- $P$: Set of projects, as listed in columns ["P1", "P2", "P3", "P4", "P5", "P6"] of table_id "file_0_view_0"

##### Parameters

- $c_{ij}$: Cost of assigning manager $i$ to project $j$, from table_id "file_0_view_0", row identifier $i$ (column "Unnamed: 0"), column $j$ (one of ["P1", "P2", "P3", "P4", "P5", "P6"])

##### Decision Variables

- $x_{ij} = \begin{cases} 1 & \text{if manager } i \text{ is assigned to project } j \\ 0 & \text{otherwise} \end{cases}$

##### Data Mapping

{
  "table_id": "file_0_view_0",
  "manager_id_column": "Unnamed: 0",
  "project_columns": ["P1", "P2", "P3", "P4", "P5", "P6"],
  "cost_parameter": "c_{ij} = \text{value at } [\text{row } i, \text{column } j] \text{ in table_id file_0_view_0}"
}