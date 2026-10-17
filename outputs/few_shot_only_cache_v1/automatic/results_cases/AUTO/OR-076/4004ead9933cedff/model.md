##### Sets

Let $I = \{W1, W2, W3, W4, W5, W6, W7, W8, W9, W10\}$ be the set of potential warehouses.

Let $J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20\}$ be the set of customers.

##### Parameters

- Fixed opening costs for each warehouse $i \in I$:
  - $f_{W1} = 2000$
  - $f_{W2} = 2500$
  - $f_{W3} = 1800$
  - $f_{W4} = 3200$
  - $f_{W5} = 1500$
  - $f_{W6} = 4000$
  - $f_{W7} = 2800$
  - $f_{W8} = 1950$
  - $f_{W9} = 3500$
  - $f_{W10} = 2200$

- Capacity of each warehouse $i \in I$:
  - $cap_{W1} = 1000$
  - $cap_{W2} = 1500$
  - $cap_{W3} = 1200$
  - $cap_{W4} = 2000$
  - $cap_{W5} = 800$
  - $cap_{W6} = 2500$
  - $cap_{W7} = 1800$
  - $cap_{W8} = 1100$
  - $cap_{W9} = 2100$
  - $cap_{W10} = 1300$

- Demand for each customer $j \in J$:
  - $d_{C1} = 800$
  - $d_{C2} = 600$
  - $d_{C3} = 500$
  - $d_{C4} = 700$
  - $d_{C5} = 450$
  - $d_{C6} = 950$
  - $d_{C7} = 350$
  - $d_{C8} = 850$
  - $d_{C9} = 400$
  - $d_{C10} = 750$
  - $d_{C11} = 900$
  - $d_{C12} = 550$
  - $d_{C13} = 650$
  - $d_{C14} = 820$
  - $d_{C15} = 480$
  - $d_{C16} = 920$
  - $d_{C17} = 320$
  - $d_{C18} = 780$
  - $d_{C19} = 520$
  - $d_{C20} = 680$

- Transportation costs $c_{ij}$ from warehouse $i$ to customer $j$ (all values in the table below):

|        | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|--------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W1     | 10 | 15 | 20 | 11 | 16 | 18 | 7  | 12 | 22 | 9   | 14  | 19  | 25  | 13  | 17  | 6   | 21  | 15  | 8   | 10  |
| W2     | 18 | 12 | 9  | 14 | 10 | 5  | 19 | 23 | 11 | 16  | 20  | 8   | 15  | 22  | 7   | 13  | 24  | 17  | 12  | 6   |
| W3     | 13 | 17 | 15 | 8  | 12 | 21 | 16 | 10 | 5  | 24  | 13  | 22  | 7   | 19  | 14  | 18  | 9   | 25  | 11  | 16  |
| W4     | 7  | 22 | 11 | 16 | 20 | 8  | 15 | 19 | 13 | 25  | 6   | 14  | 21  | 9   | 23  | 17  | 10  | 18  | 24  | 5   |
| W5     | 16 | 9  | 25 | 13 | 7  | 10 | 23 | 14 | 18 | 21  | 5   | 17  | 9   | 24  | 12  | 20  | 6   | 15  | 19  | 11  |
| W6     | 22 | 6  | 14 | 19 | 23 | 11 | 8  | 17 | 9  | 12  | 15  | 24  | 5   | 20  | 10  | 25  | 13  | 7   | 18  | 16  |
| W7     | 8  | 25 | 17 | 9  | 14 | 22 | 11 | 6  | 16 | 20  | 18  | 13  | 24  | 5   | 19  | 12  | 23  | 10  | 7   | 15  |
| W8     | 19 | 11 | 7  | 21 | 15 | 24 | 13 | 16 | 20 | 8   | 17  | 10  | 12  | 23  | 5   | 14  | 22  | 9   | 16  | 25  |
| W9     | 12 | 20 | 5  | 23 | 17 | 14 | 9  | 25 | 18 | 11  | 16  | 21  | 10  | 7   | 24  | 15  | 19  | 6   | 13  | 22  |
| W10    | 25 | 14 | 22 | 5  | 19 | 12 | 24 | 7  | 15 | 17  | 23  | 6   | 16  | 10  | 20  | 9   | 18  | 11  | 25  | 14  |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** Each customer's demand must be fully met.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse capacity:** The total amount shipped from each warehouse cannot exceed its capacity if it is opened.
   \[
   \sum_{j \in J} x_{ij} \leq cap_i \cdot y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### All parameters and data are as listed above, with all warehouses and customers included.