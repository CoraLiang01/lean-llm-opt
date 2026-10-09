##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}$

where $x_{ij} = 1$ if machine $i$ is assigned to task $j$, $0$ otherwise, and $c_{ij}$ is the cost of assigning machine $i$ to task $j$ as given below.

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Retrieved Information

Machines:  
M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12

Tasks:  
A, B, C, D, E, F, G, H, I, J, K, L

Cost matrix $c_{ij}$ (rows: machines M1–M12, columns: tasks A–L):

|        |   A   |   B   |   C   |   D   |   E   |   F   |   G   |   H   |   I   |   J   |   K   |   L   |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **M1**  | 167.4 | 98.6  | 189.4 | 119.6 | 182.0 | 145.1 | 185.4 | 94.8  | 122.3 | 123.3 | 96.1  | 90.3  |
| **M2**  | 156.2 | 88.7  | 187.3 | 124.7 | 173.2 | 144.3 | 179.0 | 91.5  | 115.1 | 119.5 | 100.1 | 88.6  |
| **M3**  | 184.3 | 121.0 | 216.6 | 140.0 | 196.2 | 168.8 | 205.6 | 114.2 | 133.3 | 144.5 | 116.0 | 107.7 |
| **M4**  | 157.9 | 92.9  | 185.1 | 120.3 | 175.1 | 146.2 | 180.8 | 86.3  | 111.6 | 115.9 | 98.1  | 91.1  |
| **M5**  | 175.6 | 103.6 | 204.5 | 130.0 | 192.8 | 157.5 | 194.2 | 106.9 | 129.9 | 134.9 | 105.8 | 98.6  |
| **M6**  | 166.8 | 107.0 | 199.2 | 130.4 | 183.6 | 159.5 | 187.0 | 98.2  | 121.3 | 126.2 | 105.9 | 101.8 |
| **M7**  | 159.7 | 93.2  | 183.8 | 113.0 | 171.9 | 139.1 | 169.6 | 85.1  | 110.0 | 116.7 | 90.6  | 85.2  |
| **M8**  | 184.8 | 115.9 | 205.1 | 138.6 | 195.4 | 160.1 | 200.2 | 108.5 | 136.9 | 140.0 | 114.6 | 103.9 |
| **M9**  | 157.3 | 86.2  | 186.0 | 113.9 | 166.2 | 136.8 | 167.5 | 78.8  | 107.4 | 114.5 | 87.2  | 78.6  |
| **M10** | 164.8 | 97.8  | 200.9 | 125.8 | 188.9 | 151.2 | 187.7 | 99.5  | 119.5 | 132.1 | 101.1 | 98.4  |
| **M11** | 164.0 | 92.2  | 186.2 | 115.7 | 174.5 | 143.0 | 175.9 | 92.3  | 114.0 | 121.2 | 93.7  | 91.2  |
| **M12** | 151.7 | 76.7  | 179.5 | 109.5 | 160.6 | 128.4 | 170.2 | 74.4  | 103.7 | 110.4 | 83.7  | 75.2  |

Where $c_{ij}$ is the cost of assigning machine $i$ (row) to task $j$ (column).

##### Decision Variables

$x_{ij} = \begin{cases}
1 & \text{if machine } i \text{ is assigned to task } j \\
0 & \text{otherwise}
\end{cases}$

##### Sets

- Machines: $i \in \{\text{M1}, \text{M2}, \ldots, \text{M12}\}$
- Tasks: $j \in \{\text{A}, \text{B}, \ldots, \text{L}\}$

##### Complete Model

Minimize:
$$
\sum_{i \in \{\text{M1},\ldots,\text{M12}\}} \sum_{j \in \{\text{A},\ldots,\text{L}\}} c_{ij} x_{ij}
$$

Subject to:
$$
\sum_{j \in \{\text{A},\ldots,\text{L}\}} x_{ij} = 1 \quad \forall i \in \{\text{M1},\ldots,\text{M12}\}
$$
$$
\sum_{i \in \{\text{M1},\ldots,\text{M12}\}} x_{ij} = 1 \quad \forall j \in \{\text{A},\ldots,\text{L}\}
$$
$$
x_{ij} \in \{0,1\} \quad \forall i,j
$$

All cost parameters and identifiers are as retrieved above.