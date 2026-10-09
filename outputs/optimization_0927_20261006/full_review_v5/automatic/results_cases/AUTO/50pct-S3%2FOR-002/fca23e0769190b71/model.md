##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

where $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

##### Retrieved Information

Managers: $M = \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

Projects: $P = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

Cost matrix $c_{ij}$ (cost for manager $i$ to project $j$):

|        |   P1   |   P2   |   P3   |   P4   |   P5   |   P6   |
|--------|--------|--------|--------|--------|--------|--------|
|  MA    | 2216   | 1911   | 1661   | 2122   | 1442   | 1442   |
|  MB    | 1100   | 1271   | 2764   | 2557   | 1036   | 1036   |
|  MC    | 2827   | 2784   | 2206   | 2216   | 2677   | 2677   |
|  MD    | 2627   | 1273   | 2610   | 1957   | 1594   | 1594   |
|  ME    | 3359   | 1003   | 2554   | 1706   | 2065   | 2065   |
|  MF    | 1579   | 2289   | 2368   | 1922   | 2740   | 2740   |

Additional fields (for reference, not used in this model):

- previous_period_P1, previous_period_P2, previous_period_P3: previous period costs per project
- previous_period_assignment_status: {Reserved, Available, Completed, Assigned}

All managers and projects are eligible for assignment as per the data.

##### Complete Mathematical Model

$\boxed{
\begin{align*}
\min \quad & \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{j \in P} x_{ij} = 1 \quad \forall i \in M \\
& \sum_{i \in M} x_{ij} = 1 \quad \forall j \in P \\
& x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P
\end{align*}
}$

with

$M = \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

$P = \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

and $c_{ij}$ as given in the table above.