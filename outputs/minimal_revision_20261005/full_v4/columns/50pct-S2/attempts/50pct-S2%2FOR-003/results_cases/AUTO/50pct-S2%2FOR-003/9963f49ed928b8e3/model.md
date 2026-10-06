##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 3. Variable domains:

$x_{mp} \in \{0,1\} \quad \forall m \in M, \; p \in P$

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
  "cost_parameter_mapping": {
    "c_{mp}": "file_0_view_0: row 'Manager' = m, column = p"
  },
  "assignment_variable": "x_{mp} = 1$ if manager $m$ is assigned to project $p$, $0$ otherwise"
}
```

- $M$ = set of managers from column "Manager" in table_id "file_0_view_0"
- $P$ = set of projects from columns ["Project 1 Cost", ..., "Project 7 Cost"] in table_id "file_0_view_0"
- $c_{mp}$ = cost for manager $m$ to complete project $p$, from table_id "file_0_view_0", row "Manager" = $m$, column = $p$
- $x_{mp}$ = binary assignment variable as defined above

**All parameters and index sets are bound to the retrieved data.**