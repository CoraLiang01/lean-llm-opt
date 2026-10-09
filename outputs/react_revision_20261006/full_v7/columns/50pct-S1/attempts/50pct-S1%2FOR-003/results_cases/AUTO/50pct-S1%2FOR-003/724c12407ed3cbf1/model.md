##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:
$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:
$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:
$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

##### Data Mapping

- $M$: Set of managers, from column "Manager" in table_id file_0_view_0 of manager_project_costs.csv.
- $P$: Set of projects, corresponding to columns {"Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"} in table_id file_0_view_0.
- $c_{ij}$: Cost parameter, value from row with "Manager" = $i$ and column = "Project $k$ Cost" for $j$ = Project $k$, in table_id file_0_view_0.
- $x_{ij}$: Binary assignment variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise.