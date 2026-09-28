##### Sets

- $I = \{A1, A2, \ldots, A15\}$: potential factory sites  
- $J = \{B1, B2, \ldots, B8\}$: distribution centers

##### Parameters

From facility_costs.csv (in source order):

| Facility | FixedCost | Capacity |
|----------|-----------|----------|
| A1       | 0         | 30       |
| A2       | 175       | 10       |
| A3       | 300       | 20       |
| A4       | 375       | 30       |
| A5       | 500       | 40       |
| A6       | 200       | 20       |
| A7       | 260       | 25       |
| A8       | 220       | 30       |
| A9       | 320       | 35       |
| A10      | 280       | 20       |
| A11      | 350       | 40       |
| A12      | 420       | 25       |
| A13      | 470       | 30       |
| A14      | 520       | 50       |
| A15      | 560       | 45       |

Let $f_i$ be the fixed cost and $u_i$ the capacity for $i\in I$.

From demand_requirements.csv (in source order):

| Destination | Demand |
|-------------|--------|
| B1          | 30     |
| B2          | 25     |
| B3          | 20     |
| B4          | 35     |
| B5          | 25     |
| B6          | 30     |
| B7          | 25     |
| B8          | 30     |

Let $d_j$ be the demand for $j\in J$.

From shipping_costs.csv (in source order):

Let $c_{ij}$ be the per-unit shipping cost from factory $i$ to distribution center $j$:

| Origin | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
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

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i$ is built, 0 otherwise, for $i\in I$
- $x_{ij} \geq 0$: quantity shipped from factory $i$ to distribution center $j$, for $i\in I$, $j\in J$

##### Objective Function

\[
\min \sum_{i\in I} f_i y_i + \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each $j\in J$,
   \[
   \sum_{i\in I} x_{ij} = d_j
   \]

2. **Factory capacity (only if built):**  
   For each $i\in I$,
   \[
   \sum_{j\in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   y_i \in \{0,1\} \quad \forall i\in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i\in I,\, j\in J
   \]

##### All identifiers and coefficients (source order):

- $I = \{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15\}$
- $J = \{B1, B2, B3, B4, B5, B6, B7, B8\}$
- $f_i$ and $u_i$ as in facility_costs.csv above
- $d_j$ as in demand_requirements.csv above
- $c_{ij}$ as in shipping_costs.csv above

This is a mixed-integer facility location model with fixed and variable costs, explicit capacity, and full demand satisfaction.