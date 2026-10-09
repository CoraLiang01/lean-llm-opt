##### Objective Function:

$\quad \min \sum_{m \in \text{Managers}} \sum_{p \in \text{Projects}} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{p \in \text{Projects}} x_{mp} = 1 \quad \forall m \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{m \in \text{Managers}} x_{mp} = 1 \quad \forall p \in \text{Projects}$

###### 3. Variable domains:

$x_{mp} \in \{0,1\} \quad \forall m \in \text{Managers}, \forall p \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Manager" of table_id file_0_view_0.
- Projects: All columns ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"] of table_id file_0_view_0.
- $c_{mp}$: Entry in column "Project k Cost" for manager $m$ and project $p$ in table_id file_0_view_0.
- $x_{mp}$: Binary variable, 1 if manager $m$ is assigned to project $p$, 0 otherwise.