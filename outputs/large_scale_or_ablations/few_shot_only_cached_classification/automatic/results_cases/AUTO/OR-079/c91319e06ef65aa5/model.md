##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether factory $i$ is constructed (binary).

##### Parameters

- $I = \{\text{A1}, \text{A2}, \ldots, \text{A15}\}$: set of potential factory sites.
- $J = \{\text{B1}, \text{B2}, \ldots, \text{B8}\}$: set of distribution centers.

- Factory fixed costs $f_i$:

| Factory | Fixed Cost | Capacity |
|---------|------------|----------|
| A1      | 0          | 30       |
| A2      | 175        | 10       |
| A3      | 300        | 20       |
| A4      | 375        | 30       |
| A5      | 500        | 40       |
| A6      | 200        | 20       |
| A7      | 260        | 25       |
| A8      | 220        | 30       |
| A9      | 320        | 35       |
| A10     | 280        | 20       |
| A11     | 350        | 40       |
| A12     | 420        | 25       |
| A13     | 470        | 30       |
| A14     | 520        | 50       |
| A15     | 560        | 45       |

- Distribution center demands $d_j$:

| Distribution Center | Demand |
|---------------------|--------|
| B1                  | 30     |
| B2                  | 25     |
| B3                  | 20     |
| B4                  | 35     |
| B5                  | 25     |
| B6                  | 30     |
| B7                  | 25     |
| B8                  | 30     |

- Shipping costs $c_{ij}$ (from factory $i$ to distribution center $j$):

|        | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|--------|----|----|----|----|----|----|----|----|
| A1     | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
| A2     | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
| A3     | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
| A4     | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
| A5     | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
| A6     | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
| A7     | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
| A8     | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
| A9     | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
| A10    | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
| A11    | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
| A12    | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
| A13    | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
| A14    | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
| A15    | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

##### Mathematical Model

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

**Subject to:**

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity and activation:**
   \[
   \sum_{j \in J} x_{ij} \leq \text{Capacity}_i \cdot y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### All Parameters (as retrieved):

- $I = \{\text{A1}, \text{A2}, \text{A3}, \text{A4}, \text{A5}, \text{A6}, \text{A7}, \text{A8}, \text{A9}, \text{A10}, \text{A11}, \text{A12}, \text{A13}, \text{A14}, \text{A15}\}$
- $J = \{\text{B1}, \text{B2}, \text{B3}, \text{B4}, \text{B5}, \text{B6}, \text{B7}, \text{B8}\}$
- $f_i$ and $\text{Capacity}_i$ as in the table above.
- $d_j$ as in the table above.
- $c_{ij}$ as in the matrix above.

This model ensures the optimal selection of factory sites and shipment plan to minimize total system cost while meeting all distribution center demands and respecting factory capacities.