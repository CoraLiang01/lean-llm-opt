##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to complete project $j$, as given in the data mapping below.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Manager" of table_id file_0_view_0.
- Projects: All columns in file_0_view_0 with names matching "Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost".
- $c_{ij}$: Value in table_id file_0_view_0 at row with "Manager" = $i$ and column "Project $k$ Cost" for project $j$ (where $j$ corresponds to project $k$).

- Decision variables: $x_{ij}$ is 1 if manager $i$ is assigned to project $j$, 0 otherwise, for all $i$ in Managers and $j$ in Projects.