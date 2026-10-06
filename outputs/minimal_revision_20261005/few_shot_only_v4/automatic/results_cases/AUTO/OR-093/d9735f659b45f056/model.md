##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Data Mapping

- Machines (rows): $i \in \{1,\ldots,12\}$, corresponding to Machine labels:
  - $i=1$: M1
  - $i=2$: M2
  - $i=3$: M3
  - $i=4$: M4
  - $i=5$: M5
  - $i=6$: M6
  - $i=7$: M7
  - $i=8$: M8
  - $i=9$: M9
  - $i=10$: M10
  - $i=11$: M11
  - $i=12$: M12

- Tasks (columns): $j \in \{1,\ldots,12\}$, corresponding to Task labels:
  - $j=1$: A
  - $j=2$: B
  - $j=3$: C
  - $j=4$: D
  - $j=5$: E
  - $j=6$: F
  - $j=7$: G
  - $j=8$: H
  - $j=9$: I
  - $j=10$: J
  - $j=11$: K
  - $j=12$: L

- Cost matrix $c_{ij}$: $c_{ij}$ is the cost of assigning Machine $i$ (M1–M12) to Task $j$ (A–L), as given in the CSV file columns and rows.

##### Variables

- $x_{ij} = \begin{cases} 1 & \text{if Machine } i \text{ is assigned to Task } j \\ 0 & \text{otherwise} \end{cases}$

##### Parameters

- $c_{ij}$: Machining cost for assigning Machine $i$ to Task $j$, as specified in the CSV file.