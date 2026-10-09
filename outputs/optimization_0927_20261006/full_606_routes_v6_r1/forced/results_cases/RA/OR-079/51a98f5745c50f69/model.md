Let:
- $y_i \in \{0,1\}$: 1 if facility $i$ is opened, 0 otherwise, for $i \in \{\text{A1},\ldots,\text{A15}\}$
- $x_{ij} \geq 0$: amount shipped from facility $i$ to distribution center $j$, for $i \in \{\text{A1},\ldots,\text{A15}\}$, $j \in \{\text{B1},\ldots,\text{B8}\}$

Parameters (from the data):

- Facility fixed costs and capacities:

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

- Distribution center demands:

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

- Shipping costs (per unit):

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

---

**Mathematical Model**

Minimize total cost:
$$
\min \left[ \sum_{i \in F} \text{FixedCost}_i \cdot y_i + \sum_{i \in F} \sum_{j \in D} \text{ShipCost}_{ij} \cdot x_{ij} \right]
$$

Subject to:

1. **Demand satisfaction at each distribution center:**
   $$
   \sum_{i \in F} x_{ij} = \text{Demand}_j \qquad \forall j \in D
   $$
   where $F = \{\text{A1},\ldots,\text{A15}\}$, $D = \{\text{B1},\ldots,\text{B8}\}$

2. **Facility capacity and open-only constraints:**
   $$
   \sum_{j \in D} x_{ij} \leq \text{Capacity}_i \cdot y_i \qquad \forall i \in F
   $$

3. **Variable domains:**
   $$
   y_i \in \{0,1\} \qquad \forall i \in F
   $$
   $$
   x_{ij} \geq 0 \qquad \forall i \in F,\, j \in D
   $$

---

**Where:**

- $\text{FixedCost}_i$ is the fixed cost of opening facility $i$ (see table above).
- $\text{Capacity}_i$ is the capacity of facility $i$ (see table above).
- $\text{Demand}_j$ is the demand at distribution center $j$ (see table above).
- $\text{ShipCost}_{ij}$ is the per-unit shipping cost from facility $i$ to distribution center $j$ (see table above).
- $y_i$ is a binary variable indicating if facility $i$ is opened.
- $x_{ij}$ is the amount shipped from facility $i$ to distribution center $j$.

All indices, coefficients, and constraints are as retrieved and in original order.