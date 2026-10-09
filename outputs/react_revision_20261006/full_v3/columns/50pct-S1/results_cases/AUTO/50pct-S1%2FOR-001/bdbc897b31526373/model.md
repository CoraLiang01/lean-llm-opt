##### Objective Function:

$\quad \min \sum_{m \in \text{Managers}} \sum_{p \in \text{Projects}} c_{mp} \, x_{mp}$

##### Constraints:

$\sum_{p \in \text{Projects}} x_{mp} = 1 \quad \forall m \in \text{Managers}$

$\sum_{m \in \text{Managers}} x_{mp} = 1 \quad \forall p \in \text{Projects}$

$x_{mp} \in \{0,1\} \quad \forall m \in \text{Managers}, \forall p \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Unnamed: 0" of table_id "file_0_view_0" in "manager_project_costs.csv"
- Projects: All column headers ["P1", "P2", "P3"] of table_id "file_0_view_0" in "manager_project_costs.csv"
- $c_{mp}$: Value at row with "Unnamed: 0" = $m$ and column $p$ in table_id "file_0_view_0" of "manager_project_costs.csv"
- $x_{mp}$: Binary decision variable, 1 if manager $m$ is assigned to project $p$, 0 otherwise