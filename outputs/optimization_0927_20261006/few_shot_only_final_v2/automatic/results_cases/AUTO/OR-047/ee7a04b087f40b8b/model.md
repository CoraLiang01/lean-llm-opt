Let:
- $i$ index platforms, with PlatformId from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ index genres, with ProductName from products.csv: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (integer, $\geq 0$)
- $v_j$ = Value of genre $j$ (from products.csv)
- $w_j$ = Weight (memory requirement) of genre $j$ (from products.csv)
- $C_i$ = Capacity of platform $i$ (from capacity.csv)

**Parameters:**

From capacity.csv:

| PlatformId | Capacity |
|------------|----------|
| 1          | 1336     |
| 2          | 1754     |
| 3          | 1617     |
| 4          | 1119     |
| 5          | 1410     |
| 6          | 627      |
| 7          | 748      |
| 8          | 1540     |
| 9          | 1292     |
| 10         | 1138     |

From products.csv:

| ProductName   | Value | Weight |
|---------------|-------|--------|
| Racing        | 28    | 393    |
| Sports        | 69    | 195    |
| Action        | 20    | 192    |
| Adventure     | 62    | 155    |
| RPG           | 58    | 500    |
| Shooter       | 11    | 156    |
| Strategy      | 73    | 317    |
| Simulation    | 43    | 694    |
| Puzzle        | 28    | 751    |
| Fighting      | 57    | 467    |
| Platformer    | 92    | 796    |
| Survival      | 66    | 146    |
| Horror        | 14    | 269    |
| Sandbox       | 49    | 246    |
| MMO           | 12    | 652    |

**Model:**

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
\]

Subject to, for each platform $i$:
\[
\sum_{j=1}^{15} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:
- $v_j$ and $w_j$ are as given in the products.csv table above, matched by ProductName.
- $C_i$ is as given in the capacity.csv table above, matched by PlatformId.

**Explicitly:**

For each $i \in \{1,2,3,4,5,6,7,8,9,10\}$ and $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$,

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \Big[ 28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} \\
&\quad + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}} \Big] \\
\text{s.t.}\quad & 393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} \\
&\quad + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq C_i \\
&\qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
\]