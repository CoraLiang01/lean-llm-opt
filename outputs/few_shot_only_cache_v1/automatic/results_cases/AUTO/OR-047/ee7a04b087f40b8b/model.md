Let $x_{ij}$ be the number of units of game genre $j$ to be listed on platform $i$.

Indices:
- $i$ indexes PlatformId $\in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ indexes ProductName (genres) $\in$ 
  {Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}

Parameters:
- $v_j$ = Value of genre $j$ (see table below)
- $w_j$ = Weight (memory requirement) of genre $j$ (see table below)
- $C_i$ = Capacity of platform $i$ (see table below)

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{15} v_j \cdot x_{ij}
\]

Subject to, for each platform $i$:
\[
\sum_{j=1}^{15} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
\]

Parameter Table:

| $j$ (ProductName) | $v_j$ (Value) | $w_j$ (Weight) |
|-------------------|--------------|---------------|
| Racing            | 28           | 393           |
| Sports            | 69           | 195           |
| Action            | 20           | 192           |
| Adventure         | 62           | 155           |
| RPG               | 58           | 500           |
| Shooter           | 11           | 156           |
| Strategy          | 73           | 317           |
| Simulation        | 43           | 694           |
| Puzzle            | 28           | 751           |
| Fighting          | 57           | 467           |
| Platformer        | 92           | 796           |
| Survival          | 66           | 146           |
| Horror            | 14           | 269           |
| Sandbox           | 49           | 246           |
| MMO               | 12           | 652           |

Platform Capacities:

| $i$ (PlatformId) | $C_i$ (Capacity) |
|------------------|------------------|
| 1                | 1336             |
| 2                | 1754             |
| 3                | 1617             |
| 4                | 1119             |
| 5                | 1410             |
| 6                | 627              |
| 7                | 748              |
| 8                | 1540             |
| 9                | 1292             |
| 10               | 1138             |

Decision variables $x_{ij}$ are nonnegative integers for all $i$ and $j$.