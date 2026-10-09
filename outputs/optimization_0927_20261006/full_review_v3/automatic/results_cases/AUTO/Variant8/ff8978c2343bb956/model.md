##### Sets and Indices

- $I = \{\text{K1}, \text{K2}, \text{K3}, \text{K4}, \text{K5}, \text{K6}\}$: set of candidate clinics, indexed by $i$.
- $J = \{\text{N1}, \text{N2}, \text{N3}, \text{N4}, \text{N5}, \text{N6}, \text{N7}, \text{N8}, \text{N9}, \text{N10}\}$: set of neighborhoods, indexed by $j$.

##### Parameters

- Neighborhood demands:
  - $d_{\text{N1}} = 30$
  - $d_{\text{N2}} = 45$
  - $d_{\text{N3}} = 25$
  - $d_{\text{N4}} = 50$
  - $d_{\text{N5}} = 40$
  - $d_{\text{N6}} = 35$
  - $d_{\text{N7}} = 55$
  - $d_{\text{N8}} = 20$
  - $d_{\text{N9}} = 60$
  - $d_{\text{N10}} = 30$

- Clinic-to-neighborhood distances $c_{ij}$:

|        | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| K1     | 2  | 3  | 9  | 10 | 11 | 12 | 13 | 14 | 15 | 16  |
| K2     | 3  | 2  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15  |
| K3     | 10 | 9  | 2  | 3  | 4  | 9  | 10 | 11 | 12 | 13  |
| K4     | 11 | 10 | 3  | 2  | 5  | 8  | 9  | 10 | 11 | 12  |
| K5     | 13 | 12 | 10 | 9  | 8  | 2  | 3  | 4  | 8  | 9   |
| K6     | 14 | 13 | 11 | 10 | 9  | 3  | 2  | 5  | 3  | 2   |

- Number of clinics to open: $p = 3$

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if clinic $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to clinic $i$, 0 otherwise, for all $i \in I$, $j \in J$.

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

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### All Parameters (as retrieved)

Neighborhood demands:
- N1: 30
- N2: 45
- N3: 25
- N4: 50
- N5: 40
- N6: 35
- N7: 55
- N8: 20
- N9: 60
- N10: 30

Clinic-to-neighborhood distances:
- K1: N1=2, N2=3, N3=9, N4=10, N5=11, N6=12, N7=13, N8=14, N9=15, N10=16
- K2: N1=3, N2=2, N3=8, N4=9, N5=10, N6=11, N7=12, N8=13, N9=14, N10=15
- K3: N1=10, N2=9, N3=2, N4=3, N5=4, N6=9, N7=10, N8=11, N9=12, N10=13
- K4: N1=11, N2=10, N3=3, N4=2, N5=5, N6=8, N7=9, N8=10, N9=11, N10=12
- K5: N1=13, N2=12, N3=10, N4=9, N5=8, N6=2, N7=3, N8=4, N9=8, N10=9
- K6: N1=14, N2=13, N3=11, N4=10, N5=9, N6=3, N7=2, N8=5, N9=3, N10=2

Number of clinics to open: 3