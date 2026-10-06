##### Objective Function:

$\quad \min \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}$

where:
- $I$ is the set of managers (from the "Manager" column in table_id: file_0_view_0)
- $J$ is the set of projects (from the columns: "Project 1 Cost", "Project 2 Cost", ..., "Project 7 Cost" in table_id: file_0_view_0)
- $c_{ij}$ is the cost for manager $i$ to complete project $j$ (from the corresponding cell in table_id: file_0_view_0)
- $x_{ij}$ is a binary variable: $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in J} x_{ij} = 1 \quad \forall i \in I$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in I, \forall j \in J$

---

##### Retrieved Information (Data Mapping)

```json
{
  "table_id": "file_0_view_0",
  "manager_index": "Manager",
  "project_cost_columns": [
    "Project 1 Cost",
    "Project 2 Cost",
    "Project 3 Cost",
    "Project 4 Cost",
    "Project 5 Cost",
    "Project 6 Cost",
    "Project 7 Cost"
  ],
  "cost_parameter": "c_{ij} = \u2018[Manager]\u2019 row, \u2018Project k Cost\u2019 column, table_id: file_0_view_0"
}
```

- $I$ = all unique values in "Manager" column of table_id: file_0_view_0
- $J$ = all columns listed in "project_cost_columns" above
- $c_{ij}$ = value at row $i$ (Manager), column $j$ (Project k Cost), in table_id: file_0_view_0