##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $I = \{$A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15$\}$ (factories)
- $J = \{$B1, B2, B3, B4, B5, B6, B7, B8$\}$ (distribution centers)

- Factory fixed costs and capacities:

| Factory | FixedCost | Capacity |
|---------|-----------|----------|
| A1      | 0         | 30       |
| A2      | 175       | 10       |
| A3      | 300       | 20       |
| A4      | 375       | 30       |
| A5      | 500       | 40       |
| A6      | 200       | 20       |
| A7      | 260       | 25       |
| A8      | 220       | 30       |
| A9      | 320       | 35       |
| A10     | 280       | 20       |
| A11     | 350       | 40       |
| A12     | 420       | 25       |
| A13     | 470       | 30       |
| A14     | 520       | 50       |
| A15     | 560       | 45       |

- Distribution center demands:

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
where $c_{ij}$ is the shipping cost from factory $i$ to distribution center $j$, and $f_i$ is the fixed cost of constructing factory $i$.

**Subject to:**

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   \]
   where $d_j$ is the demand at distribution center $j$.

2. **Factory capacity and activation:**
   \[
   \sum_{j \in J} x_{ij} \leq \text{Cap}_i \cdot y_i \qquad \forall i \in I
   \]
   where $\text{Cap}_i$ is the capacity of factory $i$.

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

##### All required parameters (as retrieved):

- $I = \{$A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15$\}$
- $J = \{$B1, B2, B3, B4, B5, B6, B7, B8$\}$
- $f_i$ and $\text{Cap}_i$ as in the table above.
- $d_j$ as in the table above.
- $c_{ij}$ as in the matrix above.

This model determines which factories to construct and how much to ship from each to each distribution center, minimizing total fixed and shipping costs, while meeting all demand and respecting factory capacities.