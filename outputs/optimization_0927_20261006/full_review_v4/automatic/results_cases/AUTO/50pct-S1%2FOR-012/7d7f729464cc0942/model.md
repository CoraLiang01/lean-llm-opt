Let $x_{ij}$ be the number of units of games from genre $j$ (item_name from products.csv) to be listed on platform $i$ (resource_id from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Platforms (indexed by $i$):  
  1 (resource_capacity: 1336)  
  2 (resource_capacity: 1754)  
  3 (resource_capacity: 1617)  
  4 (resource_capacity: 1119)  
  5 (resource_capacity: 1410)  
  6 (resource_capacity: 627)  
  7 (resource_capacity: 748)  
  8 (resource_capacity: 1540)  
  9 (resource_capacity: 1292)  
  10 (resource_capacity: 1138)  

- Genres (indexed by $j$):  
  Racing (item_value: 28, resource_requirement: 393)  
  Sports (69, 195)  
  Action (20, 192)  
  Adventure (62, 155)  
  RPG (58, 500)  
  Shooter (11, 156)  
  Strategy (73, 317)  
  Simulation (43, 694)  
  Puzzle (28, 751)  
  Fighting (57, 467)  
  Platformer (92, 796)  
  Survival (66, 146)  
  Horror (14, 269)  
  Sandbox (49, 246)  
  MMO (12, 652)  

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all genres}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the item_value for genre $j$.

Subject to, for each platform $i$ (resource_id):

\[
\sum_{j} r_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $r_j$ is the resource_requirement for genre $j$, and $C_i$ is the resource_capacity for platform $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Numerical Data:**

Platforms (resource_id, resource_capacity):

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

Genres (item_name, item_value, resource_requirement):

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

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \Big[28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} \\
&\quad + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} \\
&\quad + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}} \Big] \\
\text{s.t.}\quad & 393\,x_{1,\text{Racing}} + 195\,x_{1,\text{Sports}} + 192\,x_{1,\text{Action}} + 155\,x_{1,\text{Adventure}} + 500\,x_{1,\text{RPG}} \\
&\quad + 156\,x_{1,\text{Shooter}} + 317\,x_{1,\text{Strategy}} + 694\,x_{1,\text{Simulation}} + 751\,x_{1,\text{Puzzle}} + 467\,x_{1,\text{Fighting}} \\
&\quad + 796\,x_{1,\text{Platformer}} + 146\,x_{1,\text{Survival}} + 269\,x_{1,\text{Horror}} + 246\,x_{1,\text{Sandbox}} + 652\,x_{1,\text{MMO}} \leq 1336 \\
& \vdots \\
& 393\,x_{10,\text{Racing}} + 195\,x_{10,\text{Sports}} + 192\,x_{10,\text{Action}} + 155\,x_{10,\text{Adventure}} + 500\,x_{10,\text{RPG}} \\
&\quad + 156\,x_{10,\text{Shooter}} + 317\,x_{10,\text{Strategy}} + 694\,x_{10,\text{Simulation}} + 751\,x_{10,\text{Puzzle}} + 467\,x_{10,\text{Fighting}} \\
&\quad + 796\,x_{10,\text{Platformer}} + 146\,x_{10,\text{Survival}} + 269\,x_{10,\text{Horror}} + 246\,x_{10,\text{Sandbox}} + 652\,x_{10,\text{MMO}} \leq 1138 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{all genres above}\}
\end{align*}
\]