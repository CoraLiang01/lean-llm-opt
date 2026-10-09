Let:
- PlatformId ∈ {1,2,3,4,5,6,7,8,9,10} denote the platforms, with capacities as given in capacity.csv.
- ProductName ∈ {Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO} denote the genres, with values and memory requirements as given in products.csv.
- x_{i,j} ∈ ℤ₊ (nonnegative integers) denote the number of units of games from genre j to be listed on platform i.

Parameters:
From capacity.csv:
- Capacity for each platform:
  - Platform 1: 1336
  - Platform 2: 1754
  - Platform 3: 1617
  - Platform 4: 1119
  - Platform 5: 1410
  - Platform 6: 627
  - Platform 7: 748
  - Platform 8: 1540
  - Platform 9: 1292
  - Platform 10: 1138

From products.csv (for each genre j):
- Value_j, Weight_j:
  - Racing: Value = 28, Weight = 393
  - Sports: Value = 69, Weight = 195
  - Action: Value = 20, Weight = 192
  - Adventure: Value = 62, Weight = 155
  - RPG: Value = 58, Weight = 500
  - Shooter: Value = 11, Weight = 156
  - Strategy: Value = 73, Weight = 317
  - Simulation: Value = 43, Weight = 694
  - Puzzle: Value = 28, Weight = 751
  - Fighting: Value = 57, Weight = 467
  - Platformer: Value = 92, Weight = 796
  - Survival: Value = 66, Weight = 146
  - Horror: Value = 14, Weight = 269
  - Sandbox: Value = 49, Weight = 246
  - MMO: Value = 12, Weight = 652

Model:

Variables:
- For each platform i ∈ {1,...,10} and genre j ∈ {Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}:
    x_{i,j} ∈ {0,1,2,...}

Objective:
Maximize total value across all platforms:
\[
\text{Maximize} \quad \sum_{i=1}^{10} \sum_{j \in \text{Genres}} \text{Value}_j \cdot x_{i,j}
\]
That is,
\[
\text{Maximize} \quad
\sum_{i=1}^{10} \Big(
28\,x_{i,\text{Racing}} + 69\,x_{i,\text{Sports}} + 20\,x_{i,\text{Action}} + 62\,x_{i,\text{Adventure}} + 58\,x_{i,\text{RPG}} + 11\,x_{i,\text{Shooter}} + 73\,x_{i,\text{Strategy}} + 43\,x_{i,\text{Simulation}} + 28\,x_{i,\text{Puzzle}} + 57\,x_{i,\text{Fighting}} + 92\,x_{i,\text{Platformer}} + 66\,x_{i,\text{Survival}} + 14\,x_{i,\text{Horror}} + 49\,x_{i,\text{Sandbox}} + 12\,x_{i,\text{MMO}}
\Big)
\]

Subject to, for each platform i:

Memory capacity constraints:
\[
\sum_{j \in \text{Genres}} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i
\]
That is, for each i = 1,...,10:
\[
393\,x_{i,\text{Racing}} + 195\,x_{i,\text{Sports}} + 192\,x_{i,\text{Action}} + 155\,x_{i,\text{Adventure}} + 500\,x_{i,\text{RPG}} + 156\,x_{i,\text{Shooter}} + 317\,x_{i,\text{Strategy}} + 694\,x_{i,\text{Simulation}} + 751\,x_{i,\text{Puzzle}} + 467\,x_{i,\text{Fighting}} + 796\,x_{i,\text{Platformer}} + 146\,x_{i,\text{Survival}} + 269\,x_{i,\text{Horror}} + 246\,x_{i,\text{Sandbox}} + 652\,x_{i,\text{MMO}} \leq \text{Capacity}_i
\]
with
- Capacity_1 = 1336
- Capacity_2 = 1754
- Capacity_3 = 1617
- Capacity_4 = 1119
- Capacity_5 = 1410
- Capacity_6 = 627
- Capacity_7 = 748
- Capacity_8 = 1540
- Capacity_9 = 1292
- Capacity_10 = 1138

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\; j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
\]

This model maximizes the total value of games listed on all platforms, subject to each platform's memory capacity, with integer numbers of units of each genre per platform.