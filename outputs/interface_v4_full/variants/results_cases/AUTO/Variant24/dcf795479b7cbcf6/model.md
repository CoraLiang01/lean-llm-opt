##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for $i \in I$ (candidate facilities).
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j$ is assigned to facility $i$, 0 otherwise, for $i \in I$, $j \in J$.

##### Parameters

- $I = \{\text{F1}, \text{F2}, \text{F3}, \text{F4}, \text{F5}, \text{F6}\}$ (candidate facilities)
- $J = \{\text{N1}, \text{N2}, \text{N3}, \text{N4}, \text{N5}, \text{N6}, \text{N7}, \text{N8}, \text{N9}, \text{N10}\}$ (neighborhoods)
- Demands $d_j$ for $j \in J$:
  - $d_{\text{N1}} = 28$
  - $d_{\text{N2}} = 32$
  - $d_{\text{N3}} = 44$
  - $d_{\text{N4}} = 36$
  - $d_{\text{N5}} = 52$
  - $d_{\text{N6}} = 41$
  - $d_{\text{N7}} = 25$
  - $d_{\text{N8}} = 48$
  - $d_{\text{N9}} = 55$
  - $d_{\text{N10}} = 30$
- Distances $c_{ij}$ from facility $i$ to neighborhood $j$:

|        | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| **F1** | 2  | 3  | 5  | 9  | 10 | 11 | 13 | 14 | 15 | 12  |
| **F2** | 4  | 2  | 3  | 8  | 9  | 10 | 12 | 13 | 14 | 11  |
| **F3** | 9  | 8  | 4  | 2  | 3  | 5  | 9  | 10 | 11 | 7   |
| **F4** | 10 | 9  | 6  | 3  | 2  | 3  | 8  | 9  | 10 | 6   |
| **F5** | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   |
| **F6** | 12 | 11 | 8  | 7  | 6  | 5  | 5  | 4  | 3  | 2   |

- Number of facilities to open: $p = 2$

##### Objective Function

Minimize total demand-weighted assignment distance:
$$
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
$$

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   $$
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   $$
2. **Facility opening:** Exactly $p$ facilities are opened:
   $$
   \sum_{i \in I} y_i = 2
   $$
3. **Assignment only to open facilities:** A neighborhood can only be assigned to an open facility:
   $$
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   $$
4. **Variable domains:**
   $$
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   $$

##### All parameters and data

- Facilities: $I = \{\text{F1}, \text{F2}, \text{F3}, \text{F4}, \text{F5}, \text{F6}\}$
- Neighborhoods: $J = \{\text{N1}, \text{N2}, \text{N3}, \text{N4}, \text{N5}, \text{N6}, \text{N7}, \text{N8}, \text{N9}, \text{N10}\}$
- Demands: $d_{\text{N1}} = 28$, $d_{\text{N2}} = 32$, $d_{\text{N3}} = 44$, $d_{\text{N4}} = 36$, $d_{\text{N5}} = 52$, $d_{\text{N6}} = 41$, $d_{\text{N7}} = 25$, $d_{\text{N8}} = 48$, $d_{\text{N9}} = 55$, $d_{\text{N10}} = 30$
- Distances $c_{ij}$ as in the table above
- Number of facilities to open: $p = 2$