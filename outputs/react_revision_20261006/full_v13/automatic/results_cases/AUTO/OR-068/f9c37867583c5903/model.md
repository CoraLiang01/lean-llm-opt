##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i, j$

##### Data Mapping

- $i$ indexes managers: MA, MB, MC, MD, ME, MF (from column "Unnamed: 0" in table_id "file_0_view_0")
- $j$ indexes projects: P1, P2, P3, P4, P5, P6 (from columns "P1"..."P6" in table_id "file_0_view_0")
- $c_{ij}$ is the assignment cost for manager $i$ to project $j$, from the entry in row with "Unnamed: 0" = $i$ and column $j$ in table_id "file_0_view_0"
- $x_{ij}$ is a binary variable indicating if manager $i$ is assigned to project $j$