##### Sets

- $I = \{1,2,3,4,5,6,7,8,9,10\}$: Platforms (from PlatformId)
- $J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$: Game genres (from ProductName)

##### Parameters

- $C_i$: Capacity of platform $i$
    - $C_1 = 1336$
    - $C_2 = 1754$
    - $C_3 = 1617$
    - $C_4 = 1119$
    - $C_5 = 1410$
    - $C_6 = 627$
    - $C_7 = 748$
    - $C_8 = 1540$
    - $C_9 = 1292$
    - $C_{10} = 1138$
- $v_j$: Value per unit of genre $j$
    - Racing: $28$
    - Sports: $69$
    - Action: $20$
    - Adventure: $62$
    - RPG: $58$
    - Shooter: $11$
    - Strategy: $73$
    - Simulation: $43$
    - Puzzle: $28$
    - Fighting: $57$
    - Platformer: $92$
    - Survival: $66$
    - Horror: $14$
    - Sandbox: $49$
    - MMO: $12$
- $w_j$: Memory requirement per unit of genre $j$
    - Racing: $393$
    - Sports: $195$
    - Action: $192$
    - Adventure: $155$
    - RPG: $500$
    - Shooter: $156$
    - Strategy: $317$
    - Simulation: $694$
    - Puzzle: $751$
    - Fighting: $467$
    - Platformer: $796$
    - Survival: $146$
    - Horror: $269$
    - Sandbox: $246$
    - MMO: $652$

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of games from genre $j$ to be listed on platform $i$

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

- Platform memory capacity:
    \[
    \sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i \in I
    \]
- Integer nonnegativity:
    \[
    x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
    \]

##### Data

Platforms and capacities (in source order):

| PlatformId | Capacity |
|------------|----------|
| 1          | 1336     |
| 2          | 1754     |
| 3          | 1617     |
| 4          | 1119     |
| 5          | 1410     |
| 6          | 627      |
| 7          | 748      |
| 8          | 1540     |
| 9          | 1292     |
| 10         | 1138     |

Genres, values, and weights (in source order):

| ProductName  | Value | Weight |
|--------------|-------|--------|
| Racing       | 28    | 393    |
| Sports       | 69    | 195    |
| Action       | 20    | 192    |
| Adventure    | 62    | 155    |
| RPG          | 58    | 500    |
| Shooter      | 11    | 156    |
| Strategy     | 73    | 317    |
| Simulation   | 43    | 694    |
| Puzzle       | 28    | 751    |
| Fighting     | 57    | 467    |
| Platformer   | 92    | 796    |
| Survival     | 66    | 146    |
| Horror       | 14    | 269    |
| Sandbox      | 49    | 246    |
| MMO          | 12    | 652    |