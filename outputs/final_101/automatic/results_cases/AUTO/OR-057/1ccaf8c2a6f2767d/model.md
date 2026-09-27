Let $x_{ij}$ be the number of units of game $j$ (genre) to be listed on platform $i$. All $x_{ij}$ are nonnegative integers.

Let $P$ be the set of platforms, indexed by PlatformID:
$$
P = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}
$$

Let $G$ be the set of games (genres), indexed by ProductName:
$$
G = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}
$$

Let $v_j$ be the value of game $j$ (from products.csv), and $w_j$ its memory requirement (Weight).
Let $C_i$ be the memory capacity of platform $i$ (from capacity.csv).

Parameters:

- Platform capacities:
  - $C_1 = 995$
  - $C_2 = 1143$
  - $C_3 = 949$
  - $C_4 = 969$
  - $C_5 = 1649$
  - $C_6 = 870$
  - $C_7 = 1064$
  - $C_8 = 536$
  - $C_9 = 766$
  - $C_{10} = 532$
  - $C_{11} = 1703$
  - $C_{12} = 1633$
  - $C_{13} = 1203$
  - $C_{14} = 1979$
  - $C_{15} = 1797$

- Game values and memory requirements:

| ProductName   | $v_j$ | $w_j$ |
|---------------|-------|-------|
| Racing        | 59    | 776   |
| Sports        | 83    | 573   |
| Action        | 94    | 127   |
| Adventure     | 41    | 138   |
| RPG           | 96    | 385   |
| Shooter       | 12    | 263   |
| Strategy      | 83    | 473   |
| Simulation    | 36    | 387   |
| Puzzle        | 56    | 390   |
| Fighting      | 27    | 556   |
| Platformer    | 47    | 601   |
| Survival      | 24    | 441   |
| Horror        | 14    | 603   |
| Sandbox       | 22    | 411   |
| MMO           | 17    | 652   |

Model:

Objective:
$$
\max \sum_{i \in P} \sum_{j \in G} v_j \cdot x_{ij}
$$

Subject to, for each platform $i \in P$:
$$
\sum_{j \in G} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in P
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in P,\, j \in G
$$

Where:
- $x_{ij}$: number of units of game $j$ to be listed on platform $i$ (integer, $\geq 0$)
- $v_j$: value of game $j$ (see table above)
- $w_j$: memory requirement of game $j$ (see table above)
- $C_i$: memory capacity of platform $i$ (see list above)