Let $x_{ij}$ be the number of units of games from genre $j$ to be listed on platform $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Indices:
- $i$ indexes platforms: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ indexes genres: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$

Parameters:
- $v_j$ = value per unit of genre $j$ (from products.csv)
- $w_j$ = memory requirement per unit of genre $j$ (from products.csv)
- $C_i$ = memory capacity of platform $i$ (from capacity.csv)

Data:

Platforms and Capacities:
- Platform 1: $C_1 = 1336$
- Platform 2: $C_2 = 1754$
- Platform 3: $C_3 = 1617$
- Platform 4: $C_4 = 1119$
- Platform 5: $C_5 = 1410$
- Platform 6: $C_6 = 627$
- Platform 7: $C_7 = 748$
- Platform 8: $C_8 = 1540$
- Platform 9: $C_9 = 1292$
- Platform 10: $C_{10} = 1138$

Genres, Values, and Memory Requirements:
- Racing: $v = 28$, $w = 393$
- Sports: $v = 69$, $w = 195$
- Action: $v = 20$, $w = 192$
- Adventure: $v = 62$, $w = 155$
- RPG: $v = 58$, $w = 500$
- Shooter: $v = 11$, $w = 156$
- Strategy: $v = 73$, $w = 317$
- Simulation: $v = 43$, $w = 694$
- Puzzle: $v = 28$, $w = 751$
- Fighting: $v = 57$, $w = 467$
- Platformer: $v = 92$, $w = 796$
- Survival: $v = 66$, $w = 146$
- Horror: $v = 14$, $w = 269$
- Sandbox: $v = 49$, $w = 246$
- MMO: $v = 12$, $w = 652$

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j} v_j x_{ij}
\]

Subject to:

For each platform $i$:
\[
\sum_{j} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

Integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:
- $v_j$ and $w_j$ are as listed above for each genre $j$.
- $C_i$ is as listed above for each platform $i$.
- $x_{ij}$ is the number of units of games from genre $j$ to be listed on platform $i$.