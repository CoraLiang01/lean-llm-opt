##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

$x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P$

##### Data Mapping

- $M$: Set of managers, from column "Manager" in table_id file_0_view_0 of manager_project_costs.csv
- $P$: Set of projects, from columns {"Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"} in table_id file_0_view_0
- $c_{ij}$: Cost for manager $i$ to complete project $j$, from the intersection of row "Manager" $i$ and column $j$ in table_id file_0_view_0
- $x_{ij}$: Binary variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise

- Source: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/AP_testing/AP5/manager_project_costs.csv, table_id file_0_view_0