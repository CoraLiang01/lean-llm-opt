Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms, with resource_id from 1 to 10 (see below).
- $j$ indexes genres, with item_name as below.

**Parameters:**

From capacity.csv (in source order):

| resource_id | resource_capacity |
|-------------|------------------|
| 1           | 1336             |
| 2           | 1754             |
| 3           | 1617             |
| 4           | 1119             |
| 5           | 1410             |
| 6           | 627              |
| 7           | 748              |
| 8           | 1540             |
| 9           | 1292             |
| 10          | 1138             |

From products.csv (in source order):

| item_name   | item_value | resource_requirement |
|-------------|------------|---------------------|
| Racing      | 28         | 393                 |
| Sports      | 69         | 195                 |
| Action      | 20         | 192                 |
| Adventure   | 62         | 155                 |
| RPG         | 58         | 500                 |
| Shooter     | 11         | 156                 |
| Strategy    | 73         | 317                 |
| Simulation  | 43         | 694                 |
| Puzzle      | 28         | 751                 |
| Fighting    | 57         | 467                 |
| Platformer  | 92         | 796                 |
| Survival    | 66         | 146                 |
| Horror      | 14         | 269                 |
| Sandbox     | 49         | 246                 |
| MMO         | 12         | 652                 |

**Mathematical Model:**

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value for genre $j$.

Subject to, for each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $a_j$ is the resource_requirement for genre $j$, and $c_i$ is the resource_capacity for platform $i$.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Parameter values:**

- $c_i$ (resource_capacity):

    - $c_1 = 1336$
    - $c_2 = 1754$
    - $c_3 = 1617$
    - $c_4 = 1119$
    - $c_5 = 1410$
    - $c_6 = 627$
    - $c_7 = 748$
    - $c_8 = 1540$
    - $c_9 = 1292$
    - $c_{10} = 1138$

- $v_j$ (item_value) and $a_j$ (resource_requirement):

    - Racing: $v = 28$, $a = 393$
    - Sports: $v = 69$, $a = 195$
    - Action: $v = 20$, $a = 192$
    - Adventure: $v = 62$, $a = 155$
    - RPG: $v = 58$, $a = 500$
    - Shooter: $v = 11$, $a = 156$
    - Strategy: $v = 73$, $a = 317$
    - Simulation: $v = 43$, $a = 694$
    - Puzzle: $v = 28$, $a = 751$
    - Fighting: $v = 57$, $a = 467$
    - Platformer: $v = 92$, $a = 796$
    - Survival: $v = 66$, $a = 146$
    - Horror: $v = 14$, $a = 269$
    - Sandbox: $v = 49$, $a = 246$
    - MMO: $v = 12$, $a = 652$

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \Big[28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} \\
&\quad + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}} \Big] \\
\text{s.t.}\quad & 393\,x_{1,\text{Racing}} + 195\,x_{1,\text{Sports}} + 192\,x_{1,\text{Action}} + 155\,x_{1,\text{Adventure}} + 500\,x_{1,\text{RPG}} + 156\,x_{1,\text{Shooter}} + 317\,x_{1,\text{Strategy}} \\
&\quad + 694\,x_{1,\text{Simulation}} + 751\,x_{1,\text{Puzzle}} + 467\,x_{1,\text{Fighting}} + 796\,x_{1,\text{Platformer}} + 146\,x_{1,\text{Survival}} + 269\,x_{1,\text{Horror}} + 246\,x_{1,\text{Sandbox}} + 652\,x_{1,\text{MMO}} \leq 1336 \\
& \vdots \\
& 393\,x_{10,\text{Racing}} + 195\,x_{10,\text{Sports}} + 192\,x_{10,\text{Action}} + 155\,x_{10,\text{Adventure}} + 500\,x_{10,\text{RPG}} + 156\,x_{10,\text{Shooter}} + 317\,x_{10,\text{Strategy}} \\
&\quad + 694\,x_{10,\text{Simulation}} + 751\,x_{10,\text{Puzzle}} + 467\,x_{10,\text{Fighting}} + 796\,x_{10,\text{Platformer}} + 146\,x_{10,\text{Survival}} + 269\,x_{10,\text{Horror}} + 246\,x_{10,\text{Sandbox}} + 652\,x_{10,\text{MMO}} \leq 1138 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}
\end{align*}
$$

All coefficients and identifiers are as retrieved and in original file order.