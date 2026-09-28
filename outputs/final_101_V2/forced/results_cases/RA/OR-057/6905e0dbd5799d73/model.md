Let $x_{ij}$ be the number of units of game $j$ (genre) to be listed on platform $i$. All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i$ indexes PlatformID $\in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $j$ indexes ProductName (game genres): Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO

Parameters:
- $v_j$ = Value of game $j$ (from products.csv)
- $w_j$ = Weight (memory requirement) of game $j$ (from products.csv)
- $C_i$ = Capacity of platform $i$ (from capacity.csv)

Data:

Platforms and Capacities:
- 1: 995
- 2: 1143
- 3: 949
- 4: 969
- 5: 1649
- 6: 870
- 7: 1064
- 8: 536
- 9: 766
- 10: 532
- 11: 1703
- 12: 1633
- 13: 1203
- 14: 1979
- 15: 1797

Games (Genres), Values, and Weights:
- Racing: Value 59, Weight 776
- Sports: Value 83, Weight 573
- Action: Value 94, Weight 127
- Adventure: Value 41, Weight 138
- RPG: Value 96, Weight 385
- Shooter: Value 12, Weight 263
- Strategy: Value 83, Weight 473
- Simulation: Value 36, Weight 387
- Puzzle: Value 56, Weight 390
- Fighting: Value 27, Weight 556
- Platformer: Value 47, Weight 601
- Survival: Value 24, Weight 441
- Horror: Value 14, Weight 603
- Sandbox: Value 22, Weight 411
- MMO: Value 17, Weight 652

Model:

Objective:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}} v_j x_{ij}
$$

Subject to, for each platform $i$:
$$
\sum_{j} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Where:
- $v_j$ and $w_j$ are as listed above for each genre $j$
- $C_i$ is as listed above for each platform $i$