##### Decision Variables

- $y_i \in \{0,1\}$: 1 if candidate location $i$ is opened, 0 otherwise, for $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to location $i$, 0 otherwise, for $i \in I$, $j \in J$.

##### Parameters

- $I = \{\text{L1}, \text{L2}, \text{L3}, \text{L4}, \text{L5}, \text{L6}, \text{L7}\}$ (candidate locations)
- $J = \{\text{A1}, \text{A2}, \text{A3}, \text{A4}, \text{A5}, \text{A6}, \text{A7}, \text{A8}, \text{A9}, \text{A10}, \text{A11}, \text{A12}\}$ (areas)
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

- Distances $c_{ij}$ from location $i$ to area $j$:

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

##### Retrieved Information

{
  "areas": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12"],
  "area_demand": {
    "A1": 25, "A2": 35, "A3": 40, "A4": 30, "A5": 50, "A6": 45, "A7": 20, "A8": 55, "A9": 60, "A10": 30, "A11": 42, "A12": 38
  },
  "locations": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
  "distance": {
    "L1": {"A1": 2, "A2": 3, "A3": 4, "A4": 8, "A5": 9, "A6": 10, "A7": 13, "A8": 14, "A9": 15, "A10": 12, "A11": 11, "A12": 10},
    "L2": {"A1": 3, "A2": 2, "A3": 3, "A4": 7, "A5": 8, "A6": 9, "A7": 12, "A8": 13, "A9": 14, "A10": 11, "A11": 10, "A12": 9},
    "L3": {"A1": 8, "A2": 7, "A3": 5, "A4": 2, "A5": 3, "A6": 4, "A7": 8, "A8": 9, "A9": 11, "A10": 7, "A11": 6, "A12": 7},
    "L4": {"A1": 9, "A2": 8, "A3": 6, "A4": 3, "A5": 2, "A6": 3, "A7": 7, "A8": 8, "A9": 10, "A10": 6, "A11": 5, "A12": 6},
    "L5": {"A1": 13, "A2": 12, "A3": 10, "A4": 8, "A5": 7, "A6": 6, "A7": 2, "A8": 3, "A9": 4, "A10": 5, "A11": 6, "A12": 7},
    "L6": {"A1": 14, "A2": 13, "A3": 11, "A4": 9, "A5": 8, "A6": 7, "A7": 3, "A8": 2, "A9": 3, "A10": 4, "A11": 5, "A12": 6},
    "L7": {"A1": 11, "A2": 10, "A3": 8, "A4": 7, "A5": 6, "A6": 5, "A7": 6, "A8": 5, "A9": 4, "A10": 2, "A11": 3, "A12": 2}
  },
  "number_of_facilities_to_open": 3
}