##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 3. Variable domains:

$x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Retrieved Information

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
  "cost_parameter": "c_{mp} = \u2018file_0_view_0\u2019[Manager=m, Project Cost Column=p]"
}

Where:
- $M$ is the set of managers from the "Manager" column in table_id "file_0_view_0"
- $P$ is the set of projects, corresponding to the columns ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"] in "file_0_view_0"
- $c_{mp}$ is the cost for manager $m$ to complete project $p$, given by the value in "file_0_view_0" at row with Manager $m$ and column $p$
- $x_{mp}$ is a binary variable equal to 1 if manager $m$ is assigned to project $p$, 0 otherwise