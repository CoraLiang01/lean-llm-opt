##### Sets and Indices

- $I = \{F1, F2, F3, F4, F5, F6\}$: set of candidate facilities, indexed by $i$.
- $J = \{N1, N2, N3, N4, N5, N6, N7, N8, N9, N10\}$: set of neighborhoods, indexed by $j$.

##### Parameters

- Neighborhood demands:
  - $d_{N1} = 28$
  - $d_{N2} = 32$
  - $d_{N3} = 44$
  - $d_{N4} = 36$
  - $d_{N5} = 52$
  - $d_{N6} = 41$
  - $d_{N7} = 25$
  - $d_{N8} = 48$
  - $d_{N9} = 55$
  - $d_{N10} = 30$

- Facility-to-neighborhood distances $c_{ij}$:

|        | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| F1     | 2  | 3  | 5  | 9  | 10 | 11 | 13 | 14 | 15 | 12  |
| F2     | 4  | 2  | 3  | 8  | 9  | 10 | 12 | 13 | 14 | 11  |
| F3     | 9  | 8  | 4  | 2  | 3  | 5  | 9  | 10 | 11 | 7   |
| F4     | 10 | 9  | 6  | 3  | 2  | 3  | 8  | 9  | 10 | 6   |
| F5     | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   |
| F6     | 12 | 11 | 8  | 7  | 6  | 5  | 5  | 4  | 3  | 2   |

- Number of facilities to open: $p = 2$

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 2
   \]

3. **Assignment-to-Open-Facility Linking:** Neighborhoods can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Binary Restrictions:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Retrieved Information

- Facilities: $I = \{F1, F2, F3, F4, F5, F6\}$
- Neighborhoods: $J = \{N1, N2, N3, N4, N5, N6, N7, N8, N9, N10\}$
- Demands: $d_{N1}=28$, $d_{N2}=32$, $d_{N3}=44$, $d_{N4}=36$, $d_{N5}=52$, $d_{N6}=41$, $d_{N7}=25$, $d_{N8}=48$, $d_{N9}=55$, $d_{N10}=30$
- Distances: as in the table above
- Number of facilities to open: $p=2$