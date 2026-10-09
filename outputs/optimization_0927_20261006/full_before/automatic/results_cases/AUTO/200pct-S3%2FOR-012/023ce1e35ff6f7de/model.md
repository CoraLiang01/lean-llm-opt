Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$. All variables are required to be nonnegative integers.

**Indices:**
- $i$ indexes platforms, with resource_id from capacity.csv.
- $j$ indexes game genres, with item_name from products.csv.

**Parameters:**
- $v_j$: item_value of genre $j$ (from products.csv)
- $a_j$: resource_requirement of genre $j$ (from products.csv)
- $c_i$: resource_capacity of platform $i$ (from capacity.csv)

**Data:**

Platforms (resource_id, resource_capacity):
- 1: 1336
- 2: 1754
- 3: 1617
- 4: 1119
- 5: 1410
- 6: 627
- 7: 748
- 8: 1540
- 9: 1292
- 10: 1138

Genres (item_name, item_value, resource_requirement):
- Racing: 28, 393
- Sports: 69, 195
- Action: 20, 192
- Adventure: 62, 155
- RPG: 58, 500
- Shooter: 11, 156
- Strategy: 73, 317
- Simulation: 43, 694
- Puzzle: 28, 751
- Fighting: 57, 467
- Platformer: 92, 796
- Survival: 66, 146
- Horror: 14, 269
- Sandbox: 49, 246
- MMO: 12, 652

---

**Mathematical Model**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \ldots, \text{MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$ (resource_id):

\[
\sum_{j} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]

For all $i, j$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Parameter Table (for reference):**

| resource_id ($i$) | resource_capacity ($c_i$) |
|-------------------|--------------------------|
| 1                 | 1336                     |
| 2                 | 1754                     |
| 3                 | 1617                     |
| 4                 | 1119                     |
| 5                 | 1410                     |
| 6                 | 627                      |
| 7                 | 748                      |
| 8                 | 1540                     |
| 9                 | 1292                     |
| 10                | 1138                     |

| item_name ($j$)   | item_value ($v_j$) | resource_requirement ($a_j$) |
|-------------------|--------------------|------------------------------|
| Racing            | 28                 | 393                          |
| Sports            | 69                 | 195                          |
| Action            | 20                 | 192                          |
| Adventure         | 62                 | 155                          |
| RPG               | 58                 | 500                          |
| Shooter           | 11                 | 156                          |
| Strategy          | 73                 | 317                          |
| Simulation        | 43                 | 694                          |
| Puzzle            | 28                 | 751                          |
| Fighting          | 57                 | 467                          |
| Platformer        | 92                 | 796                          |
| Survival          | 66                 | 146                          |
| Horror            | 14                 | 269                          |
| Sandbox           | 49                 | 246                          |
| MMO               | 12                 | 652                          |

---

**Decision variables:**
\[
x_{ij} = \text{number of units of genre } j \text{ to be listed on platform } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]