##### Sets and Parameters

- Facilities: $I = \{F1, F2, F3, F4, F5, F6\}$
- Neighborhoods: $J = \{N1, N2, N3, N4, N5, N6, N7, N8, N9, N10\}$
- Demands:
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
- Distances $c_{ij}$ (facility $i$, neighborhood $j$):

|        | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|--------|----|----|----|----|----|----|----|----|----|-----|
| **F1** | 2  | 3  | 5  | 9  | 10 | 11 | 13 | 14 | 15 | 12  |
| **F2** | 4  | 2  | 3  | 8  | 9  | 10 | 12 | 13 | 14 | 11  |
| **F3** | 9  | 8  | 4  | 2  | 3  | 5  | 9  | 10 | 11 | 7   |
| **F4** | 10 | 9  | 6  | 3  | 2  | 3  | 8  | 9  | 10 | 6   |
| **F5** | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   |
| **F6** | 12 | 11 | 8  | 7  | 6  | 5  | 5  | 4  | 3  | 2   |

- Number of facilities to open: $p = 2$

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if neighborhood $j \in J$ is assigned to facility $i \in I$, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} d_j \, c_{ij} \, x_{ij}
\]

##### Constraints

1. **Assignment:** Each neighborhood is assigned to exactly one facility:
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]
2. **Facility opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 2
   \]
3. **Assignment only to open facilities:** A neighborhood can only be assigned to an open facility:
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

##### Retrieved Information

{
  "Facilities": ["F1", "F2", "F3", "F4", "F5", "F6"],
  "Neighborhoods": ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10"],
  "Demand": {
    "N1": 28, "N2": 32, "N3": 44, "N4": 36, "N5": 52,
    "N6": 41, "N7": 25, "N8": 48, "N9": 55, "N10": 30
  },
  "Distance": {
    "F1": {"N1": 2, "N2": 3, "N3": 5, "N4": 9, "N5": 10, "N6": 11, "N7": 13, "N8": 14, "N9": 15, "N10": 12},
    "F2": {"N1": 4, "N2": 2, "N3": 3, "N4": 8, "N5": 9, "N6": 10, "N7": 12, "N8": 13, "N9": 14, "N10": 11},
    "F3": {"N1": 9, "N2": 8, "N3": 4, "N4": 2, "N5": 3, "N6": 5, "N7": 9, "N8": 10, "N9": 11, "N10": 7},
    "F4": {"N1": 10, "N2": 9, "N3": 6, "N4": 3, "N5": 2, "N6": 3, "N7": 8, "N8": 9, "N9": 10, "N10": 6},
    "F5": {"N1": 13, "N2": 12, "N3": 10, "N4": 8, "N5": 7, "N6": 6, "N7": 2, "N8": 3, "N9": 4, "N10": 5},
    "F6": {"N1": 12, "N2": 11, "N3": 8, "N4": 7, "N5": 6, "N6": 5, "N7": 5, "N8": 4, "N9": 3, "N10": 2}
  },
  "NumberOfFacilitiesToOpen": 2
}