**Sets:**  
- $i \in \{\text{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}\}$ (PlatformId from capacity.csv)  
- $j \in \{\text{Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}\}$ (ProductName from products.csv)  

**Parameters:**  
- $c_i$ = Capacity of platform $i$ (from capacity.csv)  
- $v_j$ = Value per unit of genre $j$ (from products.csv)  
- $w_j$ = Memory requirement per unit of genre $j$ (from products.csv)  

**Decision Variables:**  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of genre $j$ to list on platform $i$  

**Objective:**  
$$
\max \sum_{i \in \{1,2,3,4,5,6,7,8,9,10\}} \sum_{j \in \{\text{Racing, Sports, Action, Adventure, RPG, Shooter, Strategy, Simulation, Puzzle, Fighting, Platformer, Survival, Horror, Sandbox, MMO}\}} v_j x_{ij}
$$

**Subject to:**  

For each platform $i$ (PlatformId from capacity.csv):  
$$
\sum_{j} w_j x_{ij} \leq c_i
$$

For all $i, j$:  
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Numerical Data:**  

From capacity.csv:  
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

From products.csv:  
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

**Complete Mathematical Model:**  

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{15} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{15} w_j x_{ij} \leq c_i, \quad \forall i \in \{1,\ldots,10\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,15\}
\end{align*}
$$

Where:  
- $v_j$ and $w_j$ are as given in the products.csv table above, indexed in the original order.  
- $c_i$ is as given in the capacity.csv table above, indexed in the original order.  
- $x_{ij}$ is the integer number of units of genre $j$ to list on platform $i$.