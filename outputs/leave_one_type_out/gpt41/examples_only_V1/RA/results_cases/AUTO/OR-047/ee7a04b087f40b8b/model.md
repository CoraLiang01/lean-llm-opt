Let:
- Platforms: i ∈ {1,2,3,4,5,6,7,8,9,10} (from PlatformId in capacity.csv)
- Genres: j ∈ {Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO} (from ProductName in products.csv)
- x_ij = number of units of games from genre j to be listed on platform i (integer, ≥ 0)

Parameters:
From capacity.csv:
PlatformId | Capacity
1 | 1336
2 | 1754
3 | 1617
4 | 1119
5 | 1410
6 | 627
7 | 748
8 | 1540
9 | 1292
10 | 1138

From products.csv:
Genre (j) | Value_j | Weight_j
Racing | 28 | 393
Sports | 69 | 195
Action | 20 | 192
Adventure | 62 | 155
RPG | 58 | 500
Shooter | 11 | 156
Strategy | 73 | 317
Simulation | 43 | 694
Puzzle | 28 | 751
Fighting | 57 | 467
Platformer | 92 | 796
Survival | 66 | 146
Horror | 14 | 269
Sandbox | 49 | 246
MMO | 12 | 652

Model:

Variables:
x_ij ∈ {0, 1, 2, ...} for all platforms i and genres j

Objective:
Maximize
28∑_{i=1}^{10} x_{i,Racing}
+ 69∑_{i=1}^{10} x_{i,Sports}
+ 20∑_{i=1}^{10} x_{i,Action}
+ 62∑_{i=1}^{10} x_{i,Adventure}
+ 58∑_{i=1}^{10} x_{i,RPG}
+ 11∑_{i=1}^{10} x_{i,Shooter}
+ 73∑_{i=1}^{10} x_{i,Strategy}
+ 43∑_{i=1}^{10} x_{i,Simulation}
+ 28∑_{i=1}^{10} x_{i,Puzzle}
+ 57∑_{i=1}^{10} x_{i,Fighting}
+ 92∑_{i=1}^{10} x_{i,Platformer}
+ 66∑_{i=1}^{10} x_{i,Survival}
+ 14∑_{i=1}^{10} x_{i,Horror}
+ 49∑_{i=1}^{10} x_{i,Sandbox}
+ 12∑_{i=1}^{10} x_{i,MMO}

Subject to, for each platform i:

For i = 1 (PlatformId 1, Capacity 1336):
393 x_{1,Racing} + 195 x_{1,Sports} + 192 x_{1,Action} + 155 x_{1,Adventure} + 500 x_{1,RPG} + 156 x_{1,Shooter} + 317 x_{1,Strategy} + 694 x_{1,Simulation} + 751 x_{1,Puzzle} + 467 x_{1,Fighting} + 796 x_{1,Platformer} + 146 x_{1,Survival} + 269 x_{1,Horror} + 246 x_{1,Sandbox} + 652 x_{1,MMO} ≤ 1336

For i = 2 (PlatformId 2, Capacity 1754):
393 x_{2,Racing} + 195 x_{2,Sports} + 192 x_{2,Action} + 155 x_{2,Adventure} + 500 x_{2,RPG} + 156 x_{2,Shooter} + 317 x_{2,Strategy} + 694 x_{2,Simulation} + 751 x_{2,Puzzle} + 467 x_{2,Fighting} + 796 x_{2,Platformer} + 146 x_{2,Survival} + 269 x_{2,Horror} + 246 x_{2,Sandbox} + 652 x_{2,MMO} ≤ 1754

For i = 3 (PlatformId 3, Capacity 1617):
393 x_{3,Racing} + 195 x_{3,Sports} + 192 x_{3,Action} + 155 x_{3,Adventure} + 500 x_{3,RPG} + 156 x_{3,Shooter} + 317 x_{3,Strategy} + 694 x_{3,Simulation} + 751 x_{3,Puzzle} + 467 x_{3,Fighting} + 796 x_{3,Platformer} + 146 x_{3,Survival} + 269 x_{3,Horror} + 246 x_{3,Sandbox} + 652 x_{3,MMO} ≤ 1617

For i = 4 (PlatformId 4, Capacity 1119):
393 x_{4,Racing} + 195 x_{4,Sports} + 192 x_{4,Action} + 155 x_{4,Adventure} + 500 x_{4,RPG} + 156 x_{4,Shooter} + 317 x_{4,Strategy} + 694 x_{4,Simulation} + 751 x_{4,Puzzle} + 467 x_{4,Fighting} + 796 x_{4,Platformer} + 146 x_{4,Survival} + 269 x_{4,Horror} + 246 x_{4,Sandbox} + 652 x_{4,MMO} ≤ 1119

For i = 5 (PlatformId 5, Capacity 1410):
393 x_{5,Racing} + 195 x_{5,Sports} + 192 x_{5,Action} + 155 x_{5,Adventure} + 500 x_{5,RPG} + 156 x_{5,Shooter} + 317 x_{5,Strategy} + 694 x_{5,Simulation} + 751 x_{5,Puzzle} + 467 x_{5,Fighting} + 796 x_{5,Platformer} + 146 x_{5,Survival} + 269 x_{5,Horror} + 246 x_{5,Sandbox} + 652 x_{5,MMO} ≤ 1410

For i = 6 (PlatformId 6, Capacity 627):
393 x_{6,Racing} + 195 x_{6,Sports} + 192 x_{6,Action} + 155 x_{6,Adventure} + 500 x_{6,RPG} + 156 x_{6,Shooter} + 317 x_{6,Strategy} + 694 x_{6,Simulation} + 751 x_{6,Puzzle} + 467 x_{6,Fighting} + 796 x_{6,Platformer} + 146 x_{6,Survival} + 269 x_{6,Horror} + 246 x_{6,Sandbox} + 652 x_{6,MMO} ≤ 627

For i = 7 (PlatformId 7, Capacity 748):
393 x_{7,Racing} + 195 x_{7,Sports} + 192 x_{7,Action} + 155 x_{7,Adventure} + 500 x_{7,RPG} + 156 x_{7,Shooter} + 317 x_{7,Strategy} + 694 x_{7,Simulation} + 751 x_{7,Puzzle} + 467 x_{7,Fighting} + 796 x_{7,Platformer} + 146 x_{7,Survival} + 269 x_{7,Horror} + 246 x_{7,Sandbox} + 652 x_{7,MMO} ≤ 748

For i = 8 (PlatformId 8, Capacity 1540):
393 x_{8,Racing} + 195 x_{8,Sports} + 192 x_{8,Action} + 155 x_{8,Adventure} + 500 x_{8,RPG} + 156 x_{8,Shooter} + 317 x_{8,Strategy} + 694 x_{8,Simulation} + 751 x_{8,Puzzle} + 467 x_{8,Fighting} + 796 x_{8,Platformer} + 146 x_{8,Survival} + 269 x_{8,Horror} + 246 x_{8,Sandbox} + 652 x_{8,MMO} ≤ 1540

For i = 9 (PlatformId 9, Capacity 1292):
393 x_{9,Racing} + 195 x_{9,Sports} + 192 x_{9,Action} + 155 x_{9,Adventure} + 500 x_{9,RPG} + 156 x_{9,Shooter} + 317 x_{9,Strategy} + 694 x_{9,Simulation} + 751 x_{9,Puzzle} + 467 x_{9,Fighting} + 796 x_{9,Platformer} + 146 x_{9,Survival} + 269 x_{9,Horror} + 246 x_{9,Sandbox} + 652 x_{9,MMO} ≤ 1292

For i = 10 (PlatformId 10, Capacity 1138):
393 x_{10,Racing} + 195 x_{10,Sports} + 192 x_{10,Action} + 155 x_{10,Adventure} + 500 x_{10,RPG} + 156 x_{10,Shooter} + 317 x_{10,Strategy} + 694 x_{10,Simulation} + 751 x_{10,Puzzle} + 467 x_{10,Fighting} + 796 x_{10,Platformer} + 146 x_{10,Survival} + 269 x_{10,Horror} + 246 x_{10,Sandbox} + 652 x_{10,MMO} ≤ 1138

And for all i, j:
x_{i,j} ∈ {0, 1, 2, ...}

This is a complete integer programming formulation for the digital game store's platform listing problem, using all provided data and respecting all requirements.