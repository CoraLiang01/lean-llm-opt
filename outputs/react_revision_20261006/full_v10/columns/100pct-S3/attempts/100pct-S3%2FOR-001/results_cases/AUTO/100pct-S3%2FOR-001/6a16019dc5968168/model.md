##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Data Mapping

- Cost parameters $c_{ij}$ are defined as follows:
  - $i$ indexes managers: "MA", "MB", "MC" (column "Unnamed: 0" in table_id "file_0_view_0")
  - $j$ indexes projects: "P1", "P2", "P3" (columns "P1", "P2", "P3" in table_id "file_0_view_0")
  - $c_{ij}$ is the value in row $i$, column $j$ of table_id "file_0_view_0" from "manager_project_costs.csv"

- Decision variables $x_{ij}$: $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise.

- All managers and projects are included as listed in the current data; no other columns or rows are used.