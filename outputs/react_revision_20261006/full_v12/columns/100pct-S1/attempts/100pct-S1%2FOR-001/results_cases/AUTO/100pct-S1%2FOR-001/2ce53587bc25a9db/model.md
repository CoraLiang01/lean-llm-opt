##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA},\ \text{MB},\ \text{MC}\}} \sum_{j \in \{\text{P1},\ \text{P2},\ \text{P3}\}} c_{ij}\ x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1},\ \text{P2},\ \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA},\ \text{MB},\ \text{MC}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA},\ \text{MB},\ \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1},\ \text{P2},\ \text{P3}\}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA},\ \text{MB},\ \text{MC}\},\ j \in \{\text{P1},\ \text{P2},\ \text{P3}\}$

##### Data Mapping

- Cost parameters $c_{ij}$ are given by the "manager_project_costs.csv" file, table_id: file_0_view_0, with:
    - Manager index $i$ from column "Unnamed: 0" (values: MA, MB, MC)
    - Project index $j$ from columns "P1", "P2", "P3"
    - $c_{ij}$ is the value in row $i$, column $j$ of table_id: file_0_view_0

- Decision variables $x_{ij}$ indicate assignment of manager $i$ to project $j$.

- All managers and projects are included as listed in the current data.