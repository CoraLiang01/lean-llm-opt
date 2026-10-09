##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Data Mapping

- $\text{Managers}$: All unique values in the "Manager" column of table_id file_0_view_0.
- $\text{Projects}$: All project columns in table_id file_0_view_0: "Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost".
- $c_{ij}$: The value in table_id file_0_view_0 at row with "Manager" = $i$ and column = $j$.
- $x_{ij}$: Binary decision variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise.