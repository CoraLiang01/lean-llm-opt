Let $x_{ij}$ be the number of units of game $j$ (genre) to be listed on platform $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes platforms, with PlatformID as below.
- $j$ indexes games/genres, with ProductName as below.

**Parameters:**
- $v_j$: Value of one unit of game $j$ (from "Value" in products.csv)
- $w_j$: Memory requirement of one unit of game $j$ (from "Weight" in products.csv)
- $C_i$: Memory capacity of platform $i$ (from "Capacity" in capacity.csv)

**Data:**

Platforms (from capacity.csv, in source order):

| PlatformID | Capacity |
|------------|----------|
| 1          | 995      |
| 2          | 1143     |
| 3          | 949      |
| 4          | 969      |
| 5          | 1649     |
| 6          | 870      |
| 7          | 1064     |
| 8          | 536      |
| 9          | 766      |
| 10         | 532      |
| 11         | 1703     |
| 12         | 1633     |
| 13         | 1203     |
| 14         | 1979     |
| 15         | 1797     |

Games/Genres (from products.csv, in source order):

| ProductName  | Value | Weight |
|--------------|-------|--------|
| Racing       | 59    | 776    |
| Sports       | 83    | 573    |
| Action       | 94    | 127    |
| Adventure    | 41    | 138    |
| RPG          | 96    | 385    |
| Shooter      | 12    | 263    |
| Strategy     | 83    | 473    |
| Simulation   | 36    | 387    |
| Puzzle       | 56    | 390    |
| Fighting     | 27    | 556    |
| Platformer   | 47    | 601    |
| Survival     | 24    | 441    |
| Horror       | 14    | 603    |
| Sandbox      | 22    | 411    |
| MMO          | 17    | 652    |

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$ (PlatformID as above):
\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Where:**

- $v_j$ and $w_j$ are as given in the table above for each ProductName $j$.
- $C_i$ is as given in the table above for each PlatformID $i$.

**All variables and parameters use the exact identifiers and coefficients as retrieved.**