##### Objective Function:

$\quad \min \sum_{i \in \text{Manager}} \sum_{j \in \text{Project}} c_{ij} \, x_{ij}$

##### Constraints:

$\sum_{i \in \text{Manager}} x_{ij} = 1 \quad \forall j \in \text{Project}$

$\sum_{j \in \text{Project}} x_{ij} = 1 \quad \forall i \in \text{Manager}$

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Manager}, \forall j \in \text{Project}$

##### Data Mapping

- Index set $\text{Manager}$: All unique values in column "Manager" of table_id file_0_view_0.
- Index set $\text{Project}$: All columns with names matching "Project * Cost" in table_id file_0_view_0, with project $j$ corresponding to column name.
- Parameter $c_{ij}$: Value in table_id file_0_view_0 at row with "Manager" $i$ and column "Project $j$ Cost".
- Variable $x_{ij}$: Binary, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise.