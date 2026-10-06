##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

---

##### Retrieved Information

```json
{
  "table_id": "file_0_view_0",
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
    "Project 1 Cost",
    "Project 2 Cost",
    "Project 3 Cost",
    "Project 4 Cost",
    "Project 5 Cost",
    "Project 6 Cost",
    "Project 7 Cost"
  ],
  "cost_matrix": "c_{ij} = value in column j for row i in table_id file_0_view_0, with i indexed by 'Manager' and j by project cost columns as listed above"
}
```

- $M$ = set of managers = all unique values in column "Manager" of table_id file_0_view_0
- $P$ = set of projects = all columns ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"] in table_id file_0_view_0
- $c_{ij}$ = cost for manager $i$ to complete project $j$ = value in row $i$ (Manager) and column $j$ (Project k Cost) in table_id file_0_view_0
- $x_{ij}$ = 1 if manager $i$ is assigned to project $j$, 0 otherwise

**All parameters and index sets are mapped directly to the source data as specified above.**