Let $x_{ij}$ be the number of units of genre $j$ to be listed on platform $i$. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Platforms (from capacity.csv, in source order):

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

- Genres (from products.csv, in source order):

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
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the item_value for genre $j$.

Subject to, for each platform $i$ (resource_id):

\[
\sum_{j} r_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $r_j$ is the resource_requirement for genre $j$, and $c_i$ is the resource_capacity for platform $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Numerical Data:**

- For each $i$ in resource_id = 1 to 10, $c_i$ is as listed above.
- For each $j$ in item_name, $v_j$ and $r_j$ are as listed above.

**Decision variables:**

- $x_{ij}$: number of units of genre $j$ to be listed on platform $i$, integer, $\geq 0$.

**Complete Formulation:**

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \Big[ 
28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} \\
&\qquad + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} \\
&\qquad + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}} \Big] \\
\text{s.t.}\quad & 393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} \\
&\qquad + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} \\
&\qquad + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq c_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j\in\{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
\end{align*}
\]

where $c_i$ is:

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