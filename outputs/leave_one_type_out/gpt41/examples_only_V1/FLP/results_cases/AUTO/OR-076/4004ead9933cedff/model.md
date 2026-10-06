Let:
- I = {W1, W2, W3, W4, W5, W6, W7, W8, W9, W10} be the set of potential warehouse locations.
- J = {C1, C2, ..., C20} be the set of customers.

Parameters:
- Fixed annual opening costs for each warehouse (f_i):

 f = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200]  
 where f_1 = 2000 (W1), f_2 = 2500 (W2), ..., f_10 = 2200 (W10)

- Maximum service capacities for each warehouse (cap_i):

 cap = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300]  
 where cap_1 = 1000 (W1), ..., cap_10 = 1300 (W10)

- Customer demands (d_j):

 d = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680]  
 where d_1 = 800 (C1), ..., d_20 = 680 (C20)

- Variable transportation costs from warehouse i to customer j (c_{ij}):  
 The cost matrix C is:

|         | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|---------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| **W1**  | 10 | 15 | 20 | 11 | 16 | 18 | 7  | 12 | 22 | 9   | 14  | 19  | 25  | 13  | 17  | 6   | 21  | 15  | 8   | 10  |
| **W2**  | 18 | 12 | 9  | 14 | 10 | 5  | 19 | 23 | 11 | 16  | 20  | 8   | 15  | 22  | 7   | 13  | 24  | 17  | 12  | 6   |
| **W3**  | 13 | 17 | 15 | 8  | 12 | 21 | 16 | 10 | 5  | 24  | 13  | 22  | 7   | 19  | 14  | 18  | 9   | 25  | 11  | 16  |
| **W4**  | 7  | 22 | 11 | 16 | 20 | 8  | 15 | 19 | 13 | 25  | 6   | 14  | 21  | 9   | 23  | 17  | 10  | 18  | 24  | 5   |
| **W5**  | 16 | 9  | 25 | 13 | 7  | 10 | 23 | 14 | 18 | 21  | 5   | 17  | 9   | 24  | 12  | 20  | 6   | 15  | 19  | 11  |
| **W6**  | 22 | 6  | 14 | 19 | 23 | 11 | 8  | 17 | 9  | 12  | 15  | 24  | 5   | 20  | 10  | 25  | 13  | 7   | 18  | 16  |
| **W7**  | 8  | 25 | 17 | 9  | 14 | 22 | 11 | 6  | 16 | 20  | 18  | 13  | 24  | 5   | 19  | 12  | 23  | 10  | 7   | 15  |
| **W8**  | 19 | 11 | 7  | 21 | 15 | 24 | 13 | 16 | 20 | 8   | 17  | 10  | 12  | 23  | 5   | 14  | 22  | 9   | 16  | 25  |
| **W9**  | 12 | 20 | 5  | 23 | 17 | 14 | 9  | 25 | 18 | 11  | 16  | 21  | 10  | 7   | 24  | 15  | 19  | 6   | 13  | 22  |
| **W10** | 25 | 14 | 22 | 5  | 19 | 12 | 24 | 7  | 15 | 17  | 23  | 6   | 16  | 10  | 20  | 9   | 18  | 11  | 25  | 14  |

Decision Variables:
- y_i ∈ {0,1}: 1 if warehouse i is opened, 0 otherwise, for i ∈ I.
- x_{ij} ≥ 0: amount of customer j's demand served from warehouse i, for i ∈ I, j ∈ J.

Mathematical Model:

Objective:
Minimize total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Each customer's demand must be fully satisfied:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Each warehouse's total shipments cannot exceed its capacity if opened:
\[
\sum_{j \in J} x_{ij} \leq cap_i \cdot y_i \quad \forall i \in I
\]

3. Non-negativity and binary constraints:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Where:
- I = {W1, W2, ..., W10}
- J = {C1, C2, ..., C20}
- f_i, cap_i, d_j, c_{ij} as specified above.

This model selects the optimal subset of warehouses to open and allocates customer demand to minimize the sum of fixed opening and variable transportation costs, subject to warehouse capacities and full demand satisfaction.