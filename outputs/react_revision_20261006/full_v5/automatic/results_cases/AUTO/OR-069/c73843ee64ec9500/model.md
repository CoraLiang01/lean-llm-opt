##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

where $c_{ij}$ is the cost of assigning manager $i$ to project $j$, as given in the data mapping below.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Data Mapping

- Index set $\text{Managers}$: All values in column "Manager" of table_id file_0_view_0.
- Index set $\text{Projects}$: All columns of the form "Project k Cost" ($k=1,\ldots,11$) in table_id file_0_view_0.
- Cost parameter $c_{ij}$: Entry in row with "Manager" = $i$ and column "Project k Cost" = $j$ in table_id file_0_view_0, where $i$ and $j$ are as above.
- Variable $x_{ij}$: Binary, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise.

- Source: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv, table_id file_0_view_0, columns "Manager", "Project 1 Cost", ..., "Project 11 Cost".