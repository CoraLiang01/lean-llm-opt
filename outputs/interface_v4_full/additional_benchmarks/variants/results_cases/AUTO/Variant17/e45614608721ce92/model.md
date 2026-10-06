##### Sets and Indices

- $I = \{\text{L1}, \text{L2}, \text{L3}, \text{L4}, \text{L5}, \text{L6}, \text{L7}\}$: set of candidate facility locations (indexed by $i$)
- $J = \{\text{A1}, \text{A2}, \text{A3}, \text{A4}, \text{A5}, \text{A6}, \text{A7}, \text{A8}, \text{A9}, \text{A10}, \text{A11}, \text{A12}\}$: set of residential areas (indexed by $j$)

##### Parameters

- $d_j$: demand of area $j$
  - $d_{\text{A1}} = 25$
  - $d_{\text{A2}} = 35$
  - $d_{\text{A3}} = 40$
  - $d_{\text{A4}} = 30$
  - $d_{\text{A5}} = 50$
  - $d_{\text{A6}} = 45$
  - $d_{\text{A7}} = 20$
  - $d_{\text{A8}} = 55$
  - $d_{\text{A9}} = 60$
  - $d_{\text{A10}} = 30$
  - $d_{\text{A11}} = 42$
  - $d_{\text{A12}} = 38$

- $c_{ij}$: distance from location $i$ to area $j$

| $c_{ij}$ | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | A12 |
|----------|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| L1       | 2  | 3  | 4  | 8  | 9  | 10 | 13 | 14 | 15 | 12  | 11  | 10  |
| L2       | 3  | 2  | 3  | 7  | 8  | 9  | 12 | 13 | 14 | 11  | 10  | 9   |
| L3       | 8  | 7  | 5  | 2  | 3  | 4  | 8  | 9  | 11 | 7   | 6   | 7   |
| L4       | 9  | 8  | 6  | 3  | 2  | 3  | 7  | 8  | 10 | 6   | 5   | 6   |
| L5       | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   | 6   | 7   |
| L6       | 14 | 13 | 11 | 9  | 8  | 7  | 3  | 2  | 3  | 4   | 5   | 6   |
| L7       | 11 | 10 | 8  | 7  | 6  | 5  | 6  | 5  | 4  | 2   | 3   | 2   |

- $p = 3$: number of facilities to open

##### Decision Variables

- $y_i \in \{0,1\}$: $=1$ if facility location $i$ is opened, $0$ otherwise
- $x_{ij} \in \{0,1\}$: $=1$ if area $j$ is assigned to facility $i$, $0$ otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each area is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \qquad \forall j \in J
   \]

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 3
   \]

3. **Assignment to Open Facilities:** Areas can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i \qquad \forall i \in I,\, j \in J
   \]

4. **Binary Restrictions:**
   \[
   x_{ij} \in \{0,1\} \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

##### All Parameters

- $I = \{\text{L1}, \text{L2}, \text{L3}, \text{L4}, \text{L5}, \text{L6}, \text{L7}\}$
- $J = \{\text{A1}, \text{A2}, \text{A3}, \text{A4}, \text{A5}, \text{A6}, \text{A7}, \text{A8}, \text{A9}, \text{A10}, \text{A11}, \text{A12}\}$
- $d_j$ and $c_{ij}$ as above
- $p = 3$