##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \forall j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Unnamed: 1" of table_id file_0_view_0
- Projects: All columns ["P1", "P2", "P3"] of table_id file_0_view_0
- $c_{ij}$: Entry in table_id file_0_view_0 at row with "Unnamed: 1" = $i$ and column $j$
- $x_{ij}$: Binary variable indicating assignment of manager $i$ to project $j$