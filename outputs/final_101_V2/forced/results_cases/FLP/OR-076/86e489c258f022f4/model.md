##### Decision Variables

$x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- $I = \{$W1, W2, W3, W4, W5, W6, W7, W8, W9, W10$\}$ (Warehouse IDs)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20$\}$ (Customer IDs)

- Warehouse fixed opening costs ($f_i$):

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

- Warehouse capacities ($u_i$):

  - $u_{W1} = 1000$
  - $u_{W2} = 1500$
  - $u_{W3} = 1200$
  - $u_{W4} = 2000$
  - $u_{W5} = 800$
  - $u_{W6} = 2500$
  - $u_{W7} = 1800$
  - $u_{W8} = 1100$
  - $u_{W9} = 2100$
  - $u_{W10} = 1300$

- Customer demands ($d_j$):

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

- Transportation costs per unit ($c_{ij}$):

|            | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W1         | 10 | 15 | 20 | 11 | 16 | 18 | 7  | 12 | 22 | 9   | 14  | 19  | 25  | 13  | 17  | 6   | 21  | 15  | 8   | 10  |
| W2         | 18 | 12 | 9  | 14 | 10 | 5  | 19 | 23 | 11 | 16  | 20  | 8   | 15  | 22  | 7   | 13  | 24  | 17  | 12  | 6   |
| W3         | 13 | 17 | 15 | 8  | 12 | 21 | 16 | 10 | 5  | 24  | 13  | 22  | 7   | 19  | 14  | 18  | 9   | 25  | 11  | 16  |
| W4         | 7  | 22 | 11 | 16 | 20 | 8  | 15 | 19 | 13 | 25  | 6   | 14  | 21  | 9   | 23  | 17  | 10  | 18  | 24  | 5   |
| W5         | 16 | 9  | 25 | 13 | 7  | 10 | 23 | 14 | 18 | 21  | 5   | 17  | 9   | 24  | 12  | 20  | 6   | 15  | 19  | 11  |
| W6         | 22 | 6  | 14 | 19 | 23 | 11 | 8  | 17 | 9  | 12  | 15  | 24  | 5   | 20  | 10  | 25  | 13  | 7   | 18  | 16  |
| W7         | 8  | 25 | 17 | 9  | 14 | 22 | 11 | 6  | 16 | 20  | 18  | 13  | 24  | 5   | 19  | 12  | 23  | 10  | 7   | 15  |
| W8         | 19 | 11 | 7  | 21 | 15 | 24 | 13 | 16 | 20 | 8   | 17  | 10  | 12  | 23  | 5   | 14  | 22  | 9   | 16  | 25  |
| W9         | 12 | 20 | 5  | 23 | 17 | 14 | 9  | 25 | 18 | 11  | 16  | 21  | 10  | 7   | 24  | 15  | 19  | 6   | 13  | 22  |
| W10        | 25 | 14 | 22 | 5  | 19 | 12 | 24 | 7  | 15 | 17  | 23  | 6   | 16  | 10  | 20  | 9   | 18  | 11  | 25  | 14  |

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse capacity:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All parameters and data are as listed above, with full vectors and matrices preserved from the source.