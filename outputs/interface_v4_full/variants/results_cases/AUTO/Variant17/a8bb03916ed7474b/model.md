##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility location $i$ is opened, 0 otherwise, for $i \in I$.
- $x_{ij} \in \{0,1\}$: 1 if area $j$ is assigned to facility $i$, 0 otherwise, for $i \in I$, $j \in J$.

##### Parameters

- $I = \{\text{L1}, \text{L2}, \text{L3}, \text{L4}, \text{L5}, \text{L6}, \text{L7}\}$ (candidate facility locations)
- $J = \{\text{A1}, \text{A2}, \text{A3}, \text{A4}, \text{A5}, \text{A6}, \text{A7}, \text{A8}, \text{A9}, \text{A10}, \text{A11}, \text{A12}\}$ (areas)
- Demands $d_j$ for $j \in J$:

  $d_{\text{A1}} = 25$, $d_{\text{A2}} = 35$, $d_{\text{A3}} = 40$, $d_{\text{A4}} = 30$, $d_{\text{A5}} = 50$, $d_{\text{A6}} = 45$, $d_{\text{A7}} = 20$, $d_{\text{A8}} = 55$, $d_{\text{A9}} = 60$, $d_{\text{A10}} = 30$, $d_{\text{A11}} = 42$, $d_{\text{A12}} = 38$

- Assignment distances $c_{ij}$ (from location $i$ to area $j$):

|        | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 | A11 | A12 |
|--------|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| L1     | 2  | 3  | 4  | 8  | 9  | 10 | 13 | 14 | 15 | 12  | 11  | 10  |
| L2     | 3  | 2  | 3  | 7  | 8  | 9  | 12 | 13 | 14 | 11  | 10  | 9   |
| L3     | 8  | 7  | 5  | 2  | 3  | 4  | 8  | 9  | 11 | 7   | 6   | 7   |
| L4     | 9  | 8  | 6  | 3  | 2  | 3  | 7  | 8  | 10 | 6   | 5   | 6   |
| L5     | 13 | 12 | 10 | 8  | 7  | 6  | 2  | 3  | 4  | 5   | 6   | 7   |
| L6     | 14 | 13 | 11 | 9  | 8  | 7  | 3  | 2  | 3  | 4   | 5   | 6   |
| L7     | 11 | 10 | 8  | 7  | 6  | 5  | 6  | 5  | 4  | 2   | 3   | 2   |

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

2. **Facility Opening:** Exactly $p$ facilities are opened:
   \[
   \sum_{i \in I} y_i = 3
   \]

3. **Assignment-to-Open-Facility Linking:** Areas can only be assigned to open facilities:
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

4. **Variable Domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (Vectors and Matrices)

- $I = \{\text{L1}, \text{L2}, \text{L3}, \text{L4}, \text{L5}, \text{L6}, \text{L7}\}$
- $J = \{\text{A1}, \text{A2}, \text{A3}, \text{A4}, \text{A5}, \text{A6}, \text{A7}, \text{A8}, \text{A9}, \text{A10}, \text{A11}, \text{A12}\}$
- $d = [25, 35, 40, 30, 50, 45, 20, 55, 60, 30, 42, 38]$ (ordered as A1, ..., A12)
- $C = [c_{ij}]$ as the $7 \times 12$ matrix above (rows: L1-L7, columns: A1-A12)
- $p = 3$

This is the complete minimum-demand-weighted-distance $p$-median model for the given data.