##### Objective Function:

$\quad \min \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} \, x_{ij}$

where $c_{ij}$ is the cost for assigning manager $i$ to project $j$ as given in the data mapping below, and $x_{ij}$ is a binary variable indicating whether manager $i$ is assigned to project $j$.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j=1}^7 x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,7\}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i=1}^7 x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,7\}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i,j$

---

##### Retrieved Information

```json
{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_id_mapping": {
      "1": "Manager 1",
      "2": "Manager 2",
      "3": "Manager 3",
      "4": "Manager 4",
      "5": "Manager 5",
      "6": "Manager 6",
      "7": "Manager 7"
    },
    "column_id_mapping": {
      "1": "Project 1 Cost",
      "2": "Project 2 Cost",
      "3": "Project 3 Cost",
      "4": "Project 4 Cost",
      "5": "Project 5 Cost",
      "6": "Project 6 Cost",
      "7": "Project 7 Cost"
    }
  },
  "managers": [
    "Manager 1",
    "Manager 2",
    "Manager 3",
    "Manager 4",
    "Manager 5",
    "Manager 6",
    "Manager 7"
  ],
  "projects": [
    "Project 1",
    "Project 2",
    "Project 3",
    "Project 4",
    "Project 5",
    "Project 6",
    "Project 7"
  ]
}
```