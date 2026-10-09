Let:
- \( I = \{1,2,3,4,5,6,7,8,9,10\} \) be the set of PlatformIds (from capacity.csv, in order).
- \( J = \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\} \) be the set of genres (from products.csv, in order).

Parameters (from the CSVs, in order):
- For each platform \( i \in I \), let \( \text{Capacity}_i \) be:
    - 1: 1336
    - 2: 1754
    - 3: 1617
    - 4: 1119
    - 5: 1410
    - 6: 627
    - 7: 748
    - 8: 1540
    - 9: 1292
    - 10: 1138
- For each genre \( j \in J \), let \( \text{Value}_j \) and \( \text{Weight}_j \) be:

| Genre        | Value | Weight |
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

Decision variables:
- For each platform \( i \in I \) and genre \( j \in J \), let \( x_{ij} \) = number of units of games from genre \( j \) to be listed on platform \( i \).
- \( x_{ij} \in \mathbb{Z}_{\geq 0} \) (nonnegative integers).

Model:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i \in I} \sum_{j \in J} \text{Value}_j \cdot x_{ij} \\
\text{subject to} \quad
& \sum_{j \in J} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\end{align*}
\]

Explicitly, for each platform \( i \):

For \( i=1 \) (PlatformId 1, Capacity 1336):
\[
393x_{1,\text{Racing}} + 195x_{1,\text{Sports}} + 192x_{1,\text{Action}} + 155x_{1,\text{Adventure}} + 500x_{1,\text{RPG}} + 156x_{1,\text{Shooter}} + 317x_{1,\text{Strategy}} + 694x_{1,\text{Simulation}} + 751x_{1,\text{Puzzle}} + 467x_{1,\text{Fighting}} + 796x_{1,\text{Platformer}} + 146x_{1,\text{Survival}} + 269x_{1,\text{Horror}} + 246x_{1,\text{Sandbox}} + 652x_{1,\text{MMO}} \leq 1336
\]
(similarly for \( i=2,\ldots,10 \) with their respective capacities).

Objective:
\[
\text{Maximize} \quad \sum_{i=1}^{10} \Big( 28x_{i,\text{Racing}} + 69x_{i,\text{Sports}} + 20x_{i,\text{Action}} + 62x_{i,\text{Adventure}} + 58x_{i,\text{RPG}} + 11x_{i,\text{Shooter}} + 73x_{i,\text{Strategy}} + 43x_{i,\text{Simulation}} + 28x_{i,\text{Puzzle}} + 57x_{i,\text{Fighting}} + 92x_{i,\text{Platformer}} + 66x_{i,\text{Survival}} + 14x_{i,\text{Horror}} + 49x_{i,\text{Sandbox}} + 12x_{i,\text{MMO}} \Big)
\]

All variables \( x_{ij} \) are nonnegative integers.

This model maximizes the total value of games listed across all platforms, subject to each platform's memory capacity, using the explicit genre values and weights from the data.