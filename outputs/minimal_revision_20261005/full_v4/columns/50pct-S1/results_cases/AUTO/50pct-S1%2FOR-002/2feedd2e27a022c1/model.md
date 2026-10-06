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
    "row_ids": ["MA", "MB", "MC", "MD", "ME", "MF"],
    "column_ids": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "mapping": "c_{ij} = \text{cost of assigning manager } i \text{ to project } j \text{ from table file_0_view_0, with i as Unnamed: 0 and j as project column}"
  },
  "Managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "Projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}