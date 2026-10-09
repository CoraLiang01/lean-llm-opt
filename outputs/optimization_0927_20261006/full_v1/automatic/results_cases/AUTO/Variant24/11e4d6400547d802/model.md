##### Sets and Indices

- $I = \{F1, F2, F3, F4, F5, F6\}$: set of candidate facilities, indexed by $i$.
- $J = \{N1, N2, N3, N4, N5, N6, N7, N8, N9, N10\}$: set of neighborhoods, indexed by $j$.

##### Parameters

- $d_j$: demand of neighborhood $j$.

Neighborhood demands:
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

- $c_{ij}$: distance from facility $i$ to neighborhood $j$.

Facility-to-neighborhood distances:

|        | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| F1     | 2  | 3  | 5  | 9  | 10 | 11 | 13 | 14 | 15 | 12  |
| F2     | 4  | 2  | 3  | 8  | 9  | 10 | 12 | 13 | 14 | 11  |
| F3     | 9  | 8  | 4  | 2  | 3  | 5  | 9  | 10 | 11 | 7   |
| F4     | 10 | 9  | 6  | 3  | 2  | 3  | 8  | 9  | 10 | 6   |
| F5     | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   |
| F6     | 12 | 11 | 8  | 7  | 6  | 5  | 5  | 4  | 3  | 2   |

- $p = 2$: number of facilities to open.

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise.

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

##### All Parameters

Neighborhoods: $J = \{N1, N2, N3, N4, N5, N6, N7, N8, N9, N10\}$

Demands:
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

Facilities: $I = \{F1, F2, F3, F4, F5, F6\}$

Distances $c_{ij}$:

- $c_{F1,N1}=2$, $c_{F1,N2}=3$, $c_{F1,N3}=5$, $c_{F1,N4}=9$, $c_{F1,N5}=10$, $c_{F1,N6}=11$, $c_{F1,N7}=13$, $c_{F1,N8}=14$, $c_{F1,N9}=15$, $c_{F1,N10}=12$
- $c_{F2,N1}=4$, $c_{F2,N2}=2$, $c_{F2,N3}=3$, $c_{F2,N4}=8$, $c_{F2,N5}=9$, $c_{F2,N6}=10$, $c_{F2,N7}=12$, $c_{F2,N8}=13$, $c_{F2,N9}=14$, $c_{F2,N10}=11$
- $c_{F3,N1}=9$, $c_{F3,N2}=8$, $c_{F3,N3}=4$, $c_{F3,N4}=2$, $c_{F3,N5}=3$, $c_{F3,N6}=5$, $c_{F3,N7}=9$, $c_{F3,N8}=10$, $c_{F3,N9}=11$, $c_{F3,N10}=7$
- $c_{F4,N1}=10$, $c_{F4,N2}=9$, $c_{F4,N3}=6$, $c_{F4,N4}=3$, $c_{F4,N5}=2$, $c_{F4,N6}=3$, $c_{F4,N7}=8$, $c_{F4,N8}=9$, $c_{F4,N9}=10$, $c_{F4,N10}=6$
- $c_{F5,N1}=13$, $c_{F5,N2}=12$, $c_{F5,N3}=10$, $c_{F5,N4}=8$, $c_{F5,N5}=7$, $c_{F5,N6}=6$, $c_{F5,N7}=2$, $c_{F5,N8}=3$, $c_{F5,N9}=4$, $c_{F5,N10}=5$
- $c_{F6,N1}=12$, $c_{F6,N2}=11$, $c_{F6,N3}=8$, $c_{F6,N4}=7$, $c_{F6,N5}=6$, $c_{F6,N6}=5$, $c_{F6,N7}=5$, $c_{F6,N8}=4$, $c_{F6,N9}=3$, $c_{F6,N10}=2$

Number of facilities to open: $p = 2$