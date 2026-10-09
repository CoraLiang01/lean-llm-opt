##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to complete project $j$, and $x_{ij}$ is a binary variable indicating whether manager $i$ is assigned to project $j$.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Manager" of table_id file_0_view_0.
- Projects: All columns with names matching "Project * Cost" in table_id file_0_view_0.
- $c_{ij}$: Entry in table_id file_0_view_0 at row with "Manager" = $i$ and column = $j$ ("Project k Cost").
- $x_{ij}$: Assignment variable for manager $i$ and project $j$.

All sets and parameters are defined exactly as in the current CSV data.