Let:
- I = {1, 2, ..., 15} be the set of PlatformIDs (from capacity.csv, in order).
- J = {"Racing", "Sports", "Action", "Adventure", "RPG", "Shooter", "Strategy", "Simulation", "Puzzle", "Fighting", "Platformer", "Survival", "Horror", "Sandbox", "MMO"} be the set of ProductNames (from products.csv, in order).

Parameters:
- Capacity_i: memory capacity of platform i (from capacity.csv).
- Value_j: value of game j (from products.csv).
- Weight_j: memory requirement of game j (from products.csv).

Decision variables:
- x_ij: integer number of units of game j to be listed on platform i, for all i in I, j in J. (x_ij ≥ 0 and integer)

Data:
From capacity.csv (in order):
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

From products.csv (in order):
| ProductName | Value | Weight |
|-------------|-------|--------|
| Racing      | 59    | 776    |
| Sports      | 83    | 573    |
| Action      | 94    | 127    |
| Adventure   | 41    | 138    |
| RPG         | 96    | 385    |
| Shooter     | 12    | 263    |
| Strategy    | 83    | 473    |
| Simulation  | 36    | 387    |
| Puzzle      | 56    | 390    |
| Fighting    | 27    | 556    |
| Platformer  | 47    | 601    |
| Survival    | 24    | 441    |
| Horror      | 14    | 603    |
| Sandbox     | 22    | 411    |
| MMO         | 17    | 652    |

Mathematical Model:

Maximize total value:
\[
\text{Maximize} \quad Z = \sum_{i \in I} \sum_{j \in J} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as above.

Subject to, for each platform i (in source order):
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Weight_j and Capacity_i are as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Explicitly, for each platform (i = 1 to 15, in order):

For i = 1 (PlatformID 1, Capacity 995):
\[
776x_{1,\text{Racing}} + 573x_{1,\text{Sports}} + 127x_{1,\text{Action}} + 138x_{1,\text{Adventure}} + 385x_{1,\text{RPG}} + 263x_{1,\text{Shooter}} + 473x_{1,\text{Strategy}} + 387x_{1,\text{Simulation}} + 390x_{1,\text{Puzzle}} + 556x_{1,\text{Fighting}} + 601x_{1,\text{Platformer}} + 441x_{1,\text{Survival}} + 603x_{1,\text{Horror}} + 411x_{1,\text{Sandbox}} + 652x_{1,\text{MMO}} \leq 995
\]
... (repeat for i = 2 to 15, using the corresponding Capacity_i)

Objective (fully expanded):
\[
\text{Maximize} \quad Z = \sum_{i=1}^{15} \Big(
59x_{i,\text{Racing}} + 83x_{i,\text{Sports}} + 94x_{i,\text{Action}} + 41x_{i,\text{Adventure}} + 96x_{i,\text{RPG}} + 12x_{i,\text{Shooter}} + 83x_{i,\text{Strategy}} + 36x_{i,\text{Simulation}} + 56x_{i,\text{Puzzle}} + 27x_{i,\text{Fighting}} + 47x_{i,\text{Platformer}} + 24x_{i,\text{Survival}} + 14x_{i,\text{Horror}} + 22x_{i,\text{Sandbox}} + 17x_{i,\text{MMO}}
\Big)
\]

Summary:
- Decision variables: x_ij ≥ 0, integer, for all PlatformID i and ProductName j.
- Objective: maximize total value of all games listed across all platforms.
- Constraints: for each platform, total memory used by all listed games does not exceed its capacity.
- All coefficients and bounds are taken directly from the supplied CSVs, in their original order.