##### Sets

- $I = \{A1, A2, \ldots, A15\}$: set of candidate factories
- $J = \{B1, B2, \ldots, B8\}$: set of distribution centers

##### Parameters

- Fixed costs and capacities (from facility_costs.csv):

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

  Let $f_i$ be the fixed cost and $K_i$ the capacity for factory $i \in I$.

- Demand at each distribution center (from demand_requirements.csv):

  | Distribution Center | Demand |
  |--------------------|--------|
  | B1                 | 30     |
  | B2                 | 25     |
  | B3                 | 20     |
  | B4                 | 35     |
  | B5                 | 25     |
  | B6                 | 30     |
  | B7                 | 25     |
  | B8                 | 30     |

  Let $d_j$ be the demand for center $j \in J$.

- Shipping costs per unit (from shipping_costs.csv):

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

  Let $c_{ij}$ be the shipping cost per unit from factory $i$ to center $j$.

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i$ is constructed, 0 otherwise, for all $i \in I$.
- $x_{ij} \geq 0$: quantity shipped from factory $i$ to distribution center $j$, for all $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity (only if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters (explicit values)

- $I = \{$A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15$\}$
- $J = \{$B1, B2, B3, B4, B5, B6, B7, B8$\}$

- $f_i$ and $K_i$:

  - $f_{A1} = 0$, $K_{A1} = 30$
  - $f_{A2} = 175$, $K_{A2} = 10$
  - $f_{A3} = 300$, $K_{A3} = 20$
  - $f_{A4} = 375$, $K_{A4} = 30$
  - $f_{A5} = 500$, $K_{A5} = 40$
  - $f_{A6} = 200$, $K_{A6} = 20$
  - $f_{A7} = 260$, $K_{A7} = 25$
  - $f_{A8} = 220$, $K_{A8} = 30$
  - $f_{A9} = 320$, $K_{A9} = 35$
  - $f_{A10} = 280$, $K_{A10} = 20$
  - $f_{A11} = 350$, $K_{A11} = 40$
  - $f_{A12} = 420$, $K_{A12} = 25$
  - $f_{A13} = 470$, $K_{A13} = 30$
  - $f_{A14} = 520$, $K_{A14} = 50$
  - $f_{A15} = 560$, $K_{A15} = 45$

- $d_j$:

  - $d_{B1} = 30$
  - $d_{B2} = 25$
  - $d_{B3} = 20$
  - $d_{B4} = 35$
  - $d_{B5} = 25$
  - $d_{B6} = 30$
  - $d_{B7} = 25$
  - $d_{B8} = 30$

- $c_{ij}$: as in the shipping cost table above.

---

**Complete Mathematical Model:**

\[
\begin{align*}
\min \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, && \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq K_i y_i, && \forall i \in I \\
& x_{ij} \geq 0, && \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, && \forall i \in I
\end{align*}
\]

Where all parameters ($f_i$, $K_i$, $d_j$, $c_{ij}$) are as specified above.