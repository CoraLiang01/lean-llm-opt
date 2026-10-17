##### Sets and Indices

- $I = \{L1, L2, L3, L4, L5, L6, L7\}$: set of candidate facility locations, indexed by $i$.
- $J = \{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12\}$: set of residential areas, indexed by $j$.

##### Parameters

- Area demands:
  - $d_{A1} = 25$
  - $d_{A2} = 35$
  - $d_{A3} = 40$
  - $d_{A4} = 30$
  - $d_{A5} = 50$
  - $d_{A6} = 45$
  - $d_{A7} = 20$
  - $d_{A8} = 55$
  - $d_{A9} = 60$
  - $d_{A10} = 30$
  - $d_{A11} = 42$
  - $d_{A12} = 38$

- Distance matrix $c_{ij}$ (distance from location $i$ to area $j$):

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

- $y_i \in \{0,1\}$: 1 if facility at location $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to facility $i$, 0 otherwise, for all $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \cdot c_{ij} \cdot x_{ij}
\]

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 3
   \]

3. **Assignment-to-Open-Facility Linking:** Areas can only be assigned to open facilities:
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

- $I = \{L1, L2, L3, L4, L5, L6, L7\}$
- $J = \{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12\}$
- $d_j$ as listed above
- $c_{ij}$ as in the table above
- $p = 3$