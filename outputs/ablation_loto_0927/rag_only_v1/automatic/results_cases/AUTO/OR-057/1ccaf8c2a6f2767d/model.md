Let:
- I = set of platforms, indexed by i (PlatformID from capacity.csv)
- J = set of games/genres, indexed by j (ProductName from products.csv)
- x_ij = number of units of game j to be listed on platform i (integer, ≥ 0)

Parameters:
- Capacity_i = memory capacity of platform i (from capacity.csv)
- Value_j = value of game j (from products.csv)
- Weight_j = memory requirement of game j (from products.csv)

Data:
capacity.csv

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

products.csv

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

Decision variables:
x_ij ∈ {0, 1, 2, ...} for all i ∈ {1,...,15}, j ∈ {Racing, Sports, ..., MMO}

Objective:
Maximize total value across all platforms:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{15} \sum_{j \in J} \text{Value}_j \cdot x_{ij}
\]

Subject to (for each platform i):
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in \{1,...,15\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
\]

Where:
- Platform indices i are from PlatformID in capacity.csv (1 to 15, as listed).
- Game indices j are from ProductName in products.csv (Racing, Sports, ..., MMO).
- Value_j and Weight_j are as in products.csv.
- Capacity_i is as in capacity.csv.

This model maximizes the total value of games listed on all platforms, subject to each platform's memory capacity, with integer numbers of units for each game-platform pair.