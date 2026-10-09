##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Data Mapping

- $M$: set of managers, from column "Manager" in table_id file_0_view_0
- $P$: set of projects, from columns "Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost" in table_id file_0_view_0
- $c_{ij}$: cost for manager $i$ to complete project $j$, from the intersection of row $i$ ("Manager") and column $j$ ("Project k Cost") in table_id file_0_view_0
- $x_{ij}$: binary variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise

- Source: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv, table_id file_0_view_0