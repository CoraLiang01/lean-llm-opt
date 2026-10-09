Let:
- I = {A1, A2, ..., A15} be the set of potential factory sites (indexed by i)
- J = {B1, B2, ..., B8} be the set of distribution centers (indexed by j)

Parameters:
- Fixed facility costs (f_i) and capacities (cap_i) for each factory i ∈ I:
  - f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560]
    - f_A1 = 0
    - f_A2 = 175
    - f_A3 = 300
    - f_A4 = 375
    - f_A5 = 500
    - f_A6 = 200
    - f_A7 = 260
    - f_A8 = 220
    - f_A9 = 320
    - f_A10 = 280
    - f_A11 = 350
    - f_A12 = 420
    - f_A13 = 470
    - f_A14 = 520
    - f_A15 = 560
  - cap = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45]
    - cap_A1 = 30
    - cap_A2 = 10
    - cap_A3 = 20
    - cap_A4 = 30
    - cap_A5 = 40
    - cap_A6 = 20
    - cap_A7 = 25
    - cap_A8 = 30
    - cap_A9 = 35
    - cap_A10 = 20
    - cap_A11 = 40
    - cap_A12 = 25
    - cap_A13 = 30
    - cap_A14 = 50
    - cap_A15 = 45

- Demand at each distribution center j ∈ J:
  - d = [30, 25, 20, 35, 25, 30, 25, 30]
    - d_B1 = 30
    - d_B2 = 25
    - d_B3 = 20
    - d_B4 = 35
    - d_B5 = 25
    - d_B6 = 30
    - d_B7 = 25
    - d_B8 = 30

- Shipping costs c_{ij} from factory i to distribution center j (matrix C, rows: A1-A15, columns: B1-B8):

|      | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|------|----|----|----|----|----|----|----|----|
| A1   | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
| A2   | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
| A3   | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
| A4   | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
| A5   | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
| A6   | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
| A7   | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
| A8   | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
| A9   | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
| A10  | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
| A11  | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
| A12  | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
| A13  | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
| A14  | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
| A15  | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

Decision Variables:
- y_i ∈ {0,1} for i ∈ I: 1 if factory i is constructed, 0 otherwise
- x_{ij} ≥ 0 for i ∈ I, j ∈ J: quantity shipped from factory i to distribution center j

Mathematical Model:

Minimize total system cost:
\[
\text{Minimize} \quad Z = \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each distribution center:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Factory capacity (only if constructed):
\[
\sum_{j \in J} x_{ij} \leq cap_i \cdot y_i \quad \forall i \in I
\]

3. Non-negativity and integrality:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Where:
- I = {A1, ..., A15}
- J = {B1, ..., B8}
- f_i, cap_i, d_j, c_{ij} as specified above.

This model ensures that only open factories can ship product, all demand is met, no factory exceeds its capacity, and the total cost (fixed + shipping) is minimized.