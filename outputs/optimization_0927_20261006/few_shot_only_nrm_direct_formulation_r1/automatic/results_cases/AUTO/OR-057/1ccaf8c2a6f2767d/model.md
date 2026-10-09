**Sets and Indices:**
- Let $i$ index PlatformID from capacity.csv: $i \in \{1,2,\ldots,15\}$
- Let $j$ index ProductName from products.csv: $j \in \{\text{Racing}, \text{Sports}, \text{Action}, \text{Adventure}, \text{RPG}, \text{Shooter}, \text{Strategy}, \text{Simulation}, \text{Puzzle}, \text{Fighting}, \text{Platformer}, \text{Survival}, \text{Horror}, \text{Sandbox}, \text{MMO}\}$

**Parameters:**
- $V_j$ = Value of genre $j$ (from products.csv)
- $W_j$ = Weight (memory requirement) of genre $j$ (from products.csv)
- $C_i$ = Capacity of platform $i$ (from capacity.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of genre $j$ to list on platform $i$

**Objective:**
\[
\max \sum_{i=1}^{15} \sum_{j=1}^{15} V_j \cdot x_{ij}
\]

**Subject to:**

For each platform $i$ (PlatformID as below):

\[
\sum_{j=1}^{15} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Data (from CSVs, in original order):**

*Platforms (capacity.csv):*

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

*Games/Genres (products.csv):*

| ProductName   | Value | Weight |
|---------------|-------|--------|
| Racing        | 59    | 776    |
| Sports        | 83    | 573    |
| Action        | 94    | 127    |
| Adventure     | 41    | 138    |
| RPG           | 96    | 385    |
| Shooter       | 12    | 263    |
| Strategy      | 83    | 473    |
| Simulation    | 36    | 387    |
| Puzzle        | 56    | 390    |
| Fighting      | 27    | 556    |
| Platformer    | 47    | 601    |
| Survival      | 24    | 441    |
| Horror        | 14    | 603    |
| Sandbox       | 22    | 411    |
| MMO           | 17    | 652    |

---

**Full Mathematical Model:**

\[
\begin{align*}
\max\quad & \sum_{i=1}^{15} \sum_{j=1}^{15} V_j \cdot x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{15} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
\]

Where:

- $V_j$ and $W_j$ are as in the table above for each genre $j$.
- $C_i$ is as in the table above for each platform $i$.
- $x_{ij}$ is the integer number of units of genre $j$ to list on platform $i$.