##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $I = \{$A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15$\}$
- $J = \{$B1, B2, B3, B4, B5, B6, B7, B8$\}$

- Fixed costs $f_i$ for each factory $i$:

| Factory | FixedCost |
|---------|-----------|
| A1      | 0         |
| A2      | 175       |
| A3      | 300       |
| A4      | 375       |
| A5      | 500       |
| A6      | 200       |
| A7      | 260       |
| A8      | 220       |
| A9      | 320       |
| A10     | 280       |
| A11     | 350       |
| A12     | 420       |
| A13     | 470       |
| A14     | 520       |
| A15     | 560       |

- Demand $d_j$ for each distribution center $j$:

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

- Shipping costs $c_{ij}$ from factory $i$ to distribution center $j$:

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

- Let $M = \sum_{j \in J} d_j = 220$ (total demand), which serves as a valid big-M upper bound for each factory's total shipment.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factories can only ship if constructed (big-M linking):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Summary of Sets and Parameters

- $I = \{$A1, A2, ..., A15$\}$
- $J = \{$B1, B2, ..., B8$\}$
- $f_i$ as above
- $d_j$ as above
- $c_{ij}$ as above
- $M = 220$

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

All parameters and sets are as listed above, with all coefficients and identifiers preserved from the CSV data.