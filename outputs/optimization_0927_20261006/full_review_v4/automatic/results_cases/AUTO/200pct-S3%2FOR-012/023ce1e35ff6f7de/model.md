Let $x_{ij}$ be the number of units of genre $j$ to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

**Sets and Indices:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (platforms, identified by resource_id)
- $j \in \{$Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO$\}$ (genres, identified by item_name)

**Parameters:**

From capacity.csv (resource_id, resource_capacity):

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

From products.csv (item_name, item_value, resource_requirement):

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

Let $v_j$ be the value of genre $j$ (item_value), and $a_j$ be the memory requirement of genre $j$ (resource_requirement). Let $c_i$ be the memory capacity of platform $i$ (resource_capacity).

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j \in \text{Genres}} v_j \cdot x_{ij}
\]

**Subject to:**

**Platform memory capacity constraints:**
\[
\sum_{j \in \text{Genres}} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

**Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, \forall j
\]

---

**Parameter Table (for reference):**

- Platforms ($i$) and their capacities ($c_i$):

    1: 1336, 2: 1754, 3: 1617, 4: 1119, 5: 1410, 6: 627, 7: 748, 8: 1540, 9: 1292, 10: 1138

- Genres ($j$), values ($v_j$), and memory requirements ($a_j$):

    - Racing: $v_{Racing}=28$, $a_{Racing}=393$
    - Sports: $v_{Sports}=69$, $a_{Sports}=195$
    - Action: $v_{Action}=20$, $a_{Action}=192$
    - Adventure: $v_{Adventure}=62$, $a_{Adventure}=155$
    - RPG: $v_{RPG}=58$, $a_{RPG}=500$
    - Shooter: $v_{Shooter}=11$, $a_{Shooter}=156$
    - Strategy: $v_{Strategy}=73$, $a_{Strategy}=317$
    - Simulation: $v_{Simulation}=43$, $a_{Simulation}=694$
    - Puzzle: $v_{Puzzle}=28$, $a_{Puzzle}=751$
    - Fighting: $v_{Fighting}=57$, $a_{Fighting}=467$
    - Platformer: $v_{Platformer}=92$, $a_{Platformer}=796$
    - Survival: $v_{Survival}=66$, $a_{Survival}=146$
    - Horror: $v_{Horror}=14$, $a_{Horror}=269$
    - Sandbox: $v_{Sandbox}=49$, $a_{Sandbox}=246$
    - MMO: $v_{MMO}=12$, $a_{MMO}=652$

---

**Decision variables:**

$x_{ij}$ = number of units of genre $j$ to be listed on platform $i$, integer, $\geq 0$.

---

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j \in \{\text{Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}\}} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in \{\text{Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}\}} a_j \cdot x_{ij} \leq c_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i, \forall j
\end{align*}
\]