##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "row_index_set": ["MA", "MB", "MC"],
    "column_index_set": ["P1", "P2", "P3"],
    "mapping": "c_{ij} = \u2018Pj\u2019 column value for manager i in 'manager_project_costs.csv'"
  },
  "Managers": ["MA", "MB", "MC"],
  "Projects": ["P1", "P2", "P3"]
}