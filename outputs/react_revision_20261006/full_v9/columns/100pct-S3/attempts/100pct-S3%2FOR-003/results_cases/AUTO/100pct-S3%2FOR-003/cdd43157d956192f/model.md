##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to complete project $j$, and $x_{ij}$ is a binary variable indicating whether manager $i$ is assigned to project $j$.

##### Constraints

###### 1. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 2. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \forall j \in \text{Projects}$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "manager_column": "Manager",
    "project_columns": [
      "Project 1 Cost",
      "Project 2 Cost",
      "Project 3 Cost",
      "Project 4 Cost",
      "Project 5 Cost",
      "Project 6 Cost",
      "Project 7 Cost"
    ]
  },
  "Managers": [
    "Manager 1",
    "Manager 2",
    "Manager 3",
    "Manager 4",
    "Manager 5",
    "Manager 6",
    "Manager 7"
  ],
  "Projects": [
    "Project 1",
    "Project 2",
    "Project 3",
    "Project 4",
    "Project 5",
    "Project 6",
    "Project 7"
  ]
}