Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$. All $x_{ij}$ are nonnegative integers.

**Sets and Indices:**
- $i$ indexes platforms, with resource_id from capacity.csv.
- $j$ indexes genres, with item_name from products.csv.

**Parameters:**
- $v_j$: item_value of genre $j$ (from products.csv)
- $a_j$: resource_requirement of genre $j$ (from products.csv)
- $c_i$: resource_capacity of platform $i$ (from capacity.csv)

**Data (in source order):**

Platforms (from capacity.csv):

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

Genres (from products.csv):

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

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$ (resource_id):

\[
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Where:
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

And for all $i, j$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Where:**

- $v_j$ and $a_j$ for each genre $j$ are:

| $j$ (item_name) | $v_j$ (item_value) | $a_j$ (resource_requirement) |
|-----------------|--------------------|------------------------------|
| Racing          | 28                 | 393                          |
| Sports          | 69                 | 195                          |
| Action          | 20                 | 192                          |
| Adventure       | 62                 | 155                          |
| RPG             | 58                 | 500                          |
| Shooter         | 11                 | 156                          |
| Strategy        | 73                 | 317                          |
| Simulation      | 43                 | 694                          |
| Puzzle          | 28                 | 751                          |
| Fighting        | 57                 | 467                          |
| Platformer      | 92                 | 796                          |
| Survival        | 66                 | 146                          |
| Horror          | 14                 | 269                          |
| Sandbox         | 49                 | 246                          |
| MMO             | 12                 | 652                          |

**Decision variables:**
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$, integer and $\geq 0$.

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{15} a_j x_{ij} \leq c_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,15
\end{align*}
\]

with all coefficients and identifiers as above.