##### Decision Variables

$x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to clinic $i$, 0 otherwise, for all clinics $i \in I$, neighborhoods $j \in J$.

$y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all clinics $i \in I$.

##### Parameters

- $I = \{K1, K2, K3, K4, K5, K6\}$ (candidate clinics)
- $J = \{N1, N2, N3, N4, N5, N6, N7, N8, N9, N10\}$ (neighborhoods)
- Demands $d_j$:
  - $d_{N1} = 30$
  - $d_{N2} = 45$
  - $d_{N3} = 25$
  - $d_{N4} = 50$
  - $d_{N5} = 40$
  - $d_{N6} = 35$
  - $d_{N7} = 55$
  - $d_{N8} = 20$
  - $d_{N9} = 60$
  - $d_{N10} = 30$
- Distances $c_{ij}$ (clinic $i$ to neighborhood $j$):

|        | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| K1     | 2  | 3  | 9  | 10 | 11 | 12 | 13 | 14 | 15 | 16  |
| K2     | 3  | 2  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15  |
| K3     | 10 | 9  | 2  | 3  | 4  | 9  | 10 | 11 | 12 | 13  |
| K4     | 11 | 10 | 3  | 2  | 5  | 8  | 9  | 10 | 11 | 12  |
| K5     | 13 | 12 | 10 | 9  | 8  | 2  | 3  | 4  | 8  | 9   |
| K6     | 14 | 13 | 11 | 10 | 9  | 3  | 2  | 5  | 3  | 2   |

- Number of clinics to open: $p = 3$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one clinic:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Clinic opening:** Exactly $p$ clinics are opened:
   \[
   \sum_{i \in I} y_i = 3
   \]

3. **Assignment only to open clinics:** A neighborhood can only be assigned to an open clinic:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]

4. **Binary restrictions:**
   \[
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### All parameters and data used are as retrieved from the CSV files.