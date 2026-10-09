**Sets:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (PlatformId from capacity.csv)
- $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of platform $i$ (from capacity.csv)
- $v_j$ = Value of genre $j$ (from products.csv)
- $w_j$ = Weight (memory requirement) of genre $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of games from genre $j$ to be listed on platform $i$

**Model:**

Maximize total value:
$$
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Racing},\ldots,\text{MMO}\}} v_j \cdot x_{ij}
$$

Subject to platform memory constraints:
$$
\sum_{j \in \{\text{Racing},\ldots,\text{MMO}\}} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Parameter Values (from CSVs):**

Platform capacities ($c_i$):

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

Genre values ($v_j$) and weights ($w_j$):

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

**Complete Model:**

Maximize
$$
\sum_{i=1}^{10} \Big[ 
28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}}
\Big]
$$

Subject to, for each platform $i$:

For $i=1$:
$$
393\,x_{1,\text{Racing}} + 195\,x_{1,\text{Sports}} + 192\,x_{1,\text{Action}} + 155\,x_{1,\text{Adventure}} + 500\,x_{1,\text{RPG}} + 156\,x_{1,\text{Shooter}} + 317\,x_{1,\text{Strategy}} + 694\,x_{1,\text{Simulation}} + 751\,x_{1,\text{Puzzle}} + 467\,x_{1,\text{Fighting}} + 796\,x_{1,\text{Platformer}} + 146\,x_{1,\text{Survival}} + 269\,x_{1,\text{Horror}} + 246\,x_{1,\text{Sandbox}} + 652\,x_{1,\text{MMO}} \leq 1336
$$

For $i=2$:
$$
393\,x_{2,\text{Racing}} + 195\,x_{2,\text{Sports}} + 192\,x_{2,\text{Action}} + 155\,x_{2,\text{Adventure}} + 500\,x_{2,\text{RPG}} + 156\,x_{2,\text{Shooter}} + 317\,x_{2,\text{Strategy}} + 694\,x_{2,\text{Simulation}} + 751\,x_{2,\text{Puzzle}} + 467\,x_{2,\text{Fighting}} + 796\,x_{2,\text{Platformer}} + 146\,x_{2,\text{Survival}} + 269\,x_{2,\text{Horror}} + 246\,x_{2,\text{Sandbox}} + 652\,x_{2,\text{MMO}} \leq 1754
$$

For $i=3$:
$$
393\,x_{3,\text{Racing}} + 195\,x_{3,\text{Sports}} + 192\,x_{3,\text{Action}} + 155\,x_{3,\text{Adventure}} + 500\,x_{3,\text{RPG}} + 156\,x_{3,\text{Shooter}} + 317\,x_{3,\text{Strategy}} + 694\,x_{3,\text{Simulation}} + 751\,x_{3,\text{Puzzle}} + 467\,x_{3,\text{Fighting}} + 796\,x_{3,\text{Platformer}} + 146\,x_{3,\text{Survival}} + 269\,x_{3,\text{Horror}} + 246\,x_{3,\text{Sandbox}} + 652\,x_{3,\text{MMO}} \leq 1617
$$

For $i=4$:
$$
393\,x_{4,\text{Racing}} + 195\,x_{4,\text{Sports}} + 192\,x_{4,\text{Action}} + 155\,x_{4,\text{Adventure}} + 500\,x_{4,\text{RPG}} + 156\,x_{4,\text{Shooter}} + 317\,x_{4,\text{Strategy}} + 694\,x_{4,\text{Simulation}} + 751\,x_{4,\text{Puzzle}} + 467\,x_{4,\text{Fighting}} + 796\,x_{4,\text{Platformer}} + 146\,x_{4,\text{Survival}} + 269\,x_{4,\text{Horror}} + 246\,x_{4,\text{Sandbox}} + 652\,x_{4,\text{MMO}} \leq 1119
$$

For $i=5$:
$$
393\,x_{5,\text{Racing}} + 195\,x_{5,\text{Sports}} + 192\,x_{5,\text{Action}} + 155\,x_{5,\text{Adventure}} + 500\,x_{5,\text{RPG}} + 156\,x_{5,\text{Shooter}} + 317\,x_{5,\text{Strategy}} + 694\,x_{5,\text{Simulation}} + 751\,x_{5,\text{Puzzle}} + 467\,x_{5,\text{Fighting}} + 796\,x_{5,\text{Platformer}} + 146\,x_{5,\text{Survival}} + 269\,x_{5,\text{Horror}} + 246\,x_{5,\text{Sandbox}} + 652\,x_{5,\text{MMO}} \leq 1410
$$

For $i=6$:
$$
393\,x_{6,\text{Racing}} + 195\,x_{6,\text{Sports}} + 192\,x_{6,\text{Action}} + 155\,x_{6,\text{Adventure}} + 500\,x_{6,\text{RPG}} + 156\,x_{6,\text{Shooter}} + 317\,x_{6,\text{Strategy}} + 694\,x_{6,\text{Simulation}} + 751\,x_{6,\text{Puzzle}} + 467\,x_{6,\text{Fighting}} + 796\,x_{6,\text{Platformer}} + 146\,x_{6,\text{Survival}} + 269\,x_{6,\text{Horror}} + 246\,x_{6,\text{Sandbox}} + 652\,x_{6,\text{MMO}} \leq 627
$$

For $i=7$:
$$
393\,x_{7,\text{Racing}} + 195\,x_{7,\text{Sports}} + 192\,x_{7,\text{Action}} + 155\,x_{7,\text{Adventure}} + 500\,x_{7,\text{RPG}} + 156\,x_{7,\text{Shooter}} + 317\,x_{7,\text{Strategy}} + 694\,x_{7,\text{Simulation}} + 751\,x_{7,\text{Puzzle}} + 467\,x_{7,\text{Fighting}} + 796\,x_{7,\text{Platformer}} + 146\,x_{7,\text{Survival}} + 269\,x_{7,\text{Horror}} + 246\,x_{7,\text{Sandbox}} + 652\,x_{7,\text{MMO}} \leq 748
$$

For $i=8$:
$$
393\,x_{8,\text{Racing}} + 195\,x_{8,\text{Sports}} + 192\,x_{8,\text{Action}} + 155\,x_{8,\text{Adventure}} + 500\,x_{8,\text{RPG}} + 156\,x_{8,\text{Shooter}} + 317\,x_{8,\text{Strategy}} + 694\,x_{8,\text{Simulation}} + 751\,x_{8,\text{Puzzle}} + 467\,x_{8,\text{Fighting}} + 796\,x_{8,\text{Platformer}} + 146\,x_{8,\text{Survival}} + 269\,x_{8,\text{Horror}} + 246\,x_{8,\text{Sandbox}} + 652\,x_{8,\text{MMO}} \leq 1540
$$

For $i=9$:
$$
393\,x_{9,\text{Racing}} + 195\,x_{9,\text{Sports}} + 192\,x_{9,\text{Action}} + 155\,x_{9,\text{Adventure}} + 500\,x_{9,\text{RPG}} + 156\,x_{9,\text{Shooter}} + 317\,x_{9,\text{Strategy}} + 694\,x_{9,\text{Simulation}} + 751\,x_{9,\text{Puzzle}} + 467\,x_{9,\text{Fighting}} + 796\,x_{9,\text{Platformer}} + 146\,x_{9,\text{Survival}} + 269\,x_{9,\text{Horror}} + 246\,x_{9,\text{Sandbox}} + 652\,x_{9,\text{MMO}} \leq 1292
$$

For $i=10$:
$$
393\,x_{10,\text{Racing}} + 195\,x_{10,\text{Sports}} + 192\,x_{10,\text{Action}} + 155\,x_{10,\text{Adventure}} + 500\,x_{10,\text{RPG}} + 156\,x_{10,\text{Shooter}} + 317\,x_{10,\text{Strategy}} + 694\,x_{10,\text{Simulation}} + 751\,x_{10,\text{Puzzle}} + 467\,x_{10,\text{Fighting}} + 796\,x_{10,\text{Platformer}} + 146\,x_{10,\text{Survival}} + 269\,x_{10,\text{Horror}} + 246\,x_{10,\text{Sandbox}} + 652\,x_{10,\text{MMO}} \leq 1138
$$

And for all $i=1,\ldots,10$ and all $j$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$