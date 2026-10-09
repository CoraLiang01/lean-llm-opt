Let $x_{ij}$ be the number of units of genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

Let $i$ index platforms by resource_id from capacity.csv, and $j$ index genres by item_name from products.csv.

Let $v_j$ be the item_value of genre $j$ (from products.csv), and $a_j$ be the resource_requirement of genre $j$ (from products.csv).

Let $c_i$ be the resource_capacity of platform $i$ (from capacity.csv).

**Sets:**

- Platforms $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (resource_id)
- Genres $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (item_name)

**Parameters:**

- $v_j$ (item_value):  
  Racing: 28  
  Sports: 69  
  Action: 20  
  Adventure: 62  
  RPG: 58  
  Shooter: 11  
  Strategy: 73  
  Simulation: 43  
  Puzzle: 28  
  Fighting: 57  
  Platformer: 92  
  Survival: 66  
  Horror: 14  
  Sandbox: 49  
  MMO: 12  

- $a_j$ (resource_requirement):  
  Racing: 393  
  Sports: 195  
  Action: 192  
  Adventure: 155  
  RPG: 500  
  Shooter: 156  
  Strategy: 317  
  Simulation: 694  
  Puzzle: 751  
  Fighting: 467  
  Platformer: 796  
  Survival: 146  
  Horror: 269  
  Sandbox: 246  
  MMO: 652  

- $c_i$ (resource_capacity):  
  1: 1336  
  2: 1754  
  3: 1617  
  4: 1119  
  5: 1410  
  6: 627  
  7: 748  
  8: 1540  
  9: 1292  
  10: 1138  

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i$ and $j$

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{all genres}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each platform $i$ (resource_id):

$$
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Data Used (in source order):**

capacity.csv (resource_id, resource_capacity):

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

products.csv (item_name, item_value, resource_requirement):

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