##### Sets

- $I = \{L1, L2, L3, L4, L5, L6, L7\}$: set of candidate facility locations
- $J = \{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12\}$: set of residential areas

##### Parameters

- Area demands $d_j$:

| Area | Demand |
|------|--------|
| A1   | 25     |
| A2   | 35     |
| A3   | 40     |
| A4   | 30     |
| A5   | 50     |
| A6   | 45     |
| A7   | 20     |
| A8   | 55     |
| A9   | 60     |
| A10  | 30     |
| A11  | 42     |
| A12  | 38     |

- Distances $c_{ij}$ (from location $i$ to area $j$):

|        | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | A12 |
|--------|----|----|----|----|----|----|----|----|----|-----|------|------|
| L1     | 2  | 3  | 4  | 8  | 9  | 10 | 13 | 14 | 15 | 12  | 11   | 10   |
| L2     | 3  | 2  | 3  | 7  | 8  | 9  | 12 | 13 | 14 | 11  | 10   | 9    |
| L3     | 8  | 7  | 5  | 2  | 3  | 4  | 8  | 9  | 11 | 7   | 6    | 7    |
| L4     | 9  | 8  | 6  | 3  | 2  | 3  | 7  | 8  | 10 | 6   | 5    | 6    |
| L5     | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   | 6    | 7    |
| L6     | 14 | 13 | 11 | 9  | 8  | 7  | 3  | 2  | 3  | 4   | 5    | 6    |
| L7     | 11 | 10 | 8  | 7  | 6  | 5  | 6  | 5  | 4  | 2   | 3    | 2    |

- Number of facilities to open: $p = 3$

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility at location $i \in I$ is opened, 0 otherwise
- $x_{ij} \in \{0,1\}$: 1 if area $j \in J$ is assigned to facility $i \in I$, 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Facility opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 3
   \]

3. **Assignment only to open facilities:** An area can only be assigned to an open facility:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Binary restrictions:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All required parameters

- $I = \{L1, L2, L3, L4, L5, L6, L7\}$
- $J = \{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12\}$
- $d_j$ as listed above
- $c_{ij}$ as listed above
- $p = 3$