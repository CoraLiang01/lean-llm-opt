##### Sets

Let $I = \{F1, F2, \ldots, F15\}$ be the set of candidate plants (indexed by $i$).  
Let $J = \{C1, C2, \ldots, C15\}$ be the set of customers (indexed by $j$).

##### Parameters

From cost.csv (plants in source order):

| Plant | Fixed Cost ($f_i$) | Capacity ($cap_i$) | Per-unit transport cost $c_{ij}$ to each customer $j$ |
|-------|--------------------|--------------------|-------------------------------------------------------|
| F1    | 11250              | 101                | C1: 7.8, C2: 7.6, C3: 6.7, C4: 7.9, C5: 8.1, C6: 8.3, C7: 7.3, C8: 8.2, C9: 8.1, C10: 8.2, C11: 7.3, C12: 7.7, C13: 6.7, C14: 7.1, C15: 7.9 |
| F2    | 13480              | 124                | C1: 5.3, C2: 6, C3: 5, C4: 6.4, C5: 5.9, C6: 6.2, C7: 5.6, C8: 6.1, C9: 6.3, C10: 6.1, C11: 5, C12: 5.6, C13: 5.3, C14: 4.9, C15: 6.3 |
| F3    | 14870              | 139                | C1: 7.2, C2: 8.1, C3: 7.4, C4: 8.8, C5: 8.5, C6: 8.7, C7: 7.7, C8: 8.7, C9: 8.9, C10: 8.5, C11: 7.2, C12: 7.7, C13: 7.1, C14: 7.6, C15: 8.4 |
| F4    | 10290              | 86                 | C1: 7, C2: 7.1, C3: 6.5, C4: 7.9, C5: 7.4, C6: 7.7, C7: 6.7, C8: 7.9, C9: 7.8, C10: 7.3, C11: 6.8, C12: 7, C13: 6.5, C14: 6.7, C15: 7.6 |
| F5    | 16740              | 157                | C1: 3.5, C2: 3.8, C3: 2.9, C4: 4.3, C5: 3.6, C6: 3.9, C7: 3.2, C8: 4.3, C9: 4.5, C10: 4, C11: 3.2, C12: 4, C13: 2.9, C14: 3.4, C15: 3.9 |
| F6    | 13960              | 133                | C1: 8.2, C2: 8.6, C3: 7.9, C4: 9.5, C5: 8.5, C6: 9.3, C7: 8.5, C8: 9.4, C9: 9, C10: 9.2, C11: 8.1, C12: 8.7, C13: 7.9, C14: 8.5, C15: 9 |
| F7    | 12680              | 118                | C1: 6.9, C2: 7.6, C3: 6.8, C4: 8.4, C5: 8, C6: 8, C7: 7.6, C8: 8, C9: 8.1, C10: 7.8, C11: 6.9, C12: 7.1, C13: 7, C14: 6.9, C15: 7.5 |
| F8    | 17890              | 162                | C1: 6.9, C2: 7.8, C3: 7.1, C4: 8.7, C5: 8.6, C6: 8.2, C7: 7.2, C8: 7.9, C9: 8.4, C10: 7.9, C11: 7, C12: 7.4, C13: 6.8, C14: 7.3, C15: 8 |
| F9    | 10950              | 92                 | C1: 3.5, C2: 3.8, C3: 2.8, C4: 4.4, C5: 4.2, C6: 4.8, C7: 3.8, C8: 5, C9: 4.5, C10: 4.1, C11: 3.2, C12: 3.7, C13: 3.7, C14: 3.2, C15: 4.5 |
| F10   | 15320              | 144                | C1: 5.2, C2: 6.1, C3: 5.1, C4: 6.3, C5: 6.1, C6: 6, C7: 5.6, C8: 6.5, C9: 6.2, C10: 5.9, C11: 5.3, C12: 6.1, C13: 5.1, C14: 5.2, C15: 6.2 |
| F11   | 11830              | 107                | C1: 5.2, C2: 5.5, C3: 4.5, C4: 6.2, C5: 5.7, C6: 6.1, C7: 5.1, C8: 5.8, C9: 5.7, C10: 6.2, C11: 5.2, C12: 5.2, C13: 4.5, C14: 5.1, C15: 5.4 |
| F12   | 14110              | 129                | C1: 7.8, C2: 8.7, C3: 7.6, C4: 9, C5: 8.6, C6: 9, C7: 8.5, C8: 9.3, C9: 9.3, C10: 8.4, C11: 7.9, C12: 8.2, C13: 7.4, C14: 7.6, C15: 8.7 |
| F13   | 15970              | 151                | C1: 6.7, C2: 6.6, C3: 6.1, C4: 7.3, C5: 7.1, C6: 7.5, C7: 6.7, C8: 8, C9: 7.6, C10: 7.2, C11: 6.3, C12: 6.9, C13: 6.2, C14: 6, C15: 7.2 |
| F14   | 13140              | 113                | C1: 7.5, C2: 8.6, C3: 7.6, C4: 8.2, C5: 8, C6: 7.9, C7: 7.5, C8: 8.7, C9: 8.8, C10: 8.1, C11: 7.2, C12: 7.3, C13: 7, C14: 7, C15: 8 |
| F15   | 10580              | 85                 | C1: 5.1, C2: 5.8, C3: 4.6, C4: 5.9, C5: 6.5, C6: 5.9, C7: 5.2, C8: 7, C9: 7.1, C10: 5.9, C11: 5.1, C12: 5.8, C13: 5.4, C14: 4.9, C15: 6 |

From demand.csv (customers in source order):

| Customer | Demand ($d_j$) |
|----------|---------------|
| C1       | 83            |
| C2       | 76            |
| C3       | 91            |
| C4       | 68            |
| C5       | 104           |
| C6       | 97            |
| C7       | 88            |
| C8       | 73            |
| C9       | 109           |
| C10      | 95            |
| C11      | 82            |
| C12      | 67            |
| C13      | 113           |
| C14      | 79            |
| C15      | 92            |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if plant $i$ is built, 0 otherwise, for all $i \in I$.
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to customer $j$, for all $i \in I$, $j \in J$ (continuous).

##### Objective Function

Minimize total cost (fixed opening + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Plant capacity (only if open):**  
   For each plant $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq cap_i \cdot y_i
   $$
3. **Non-negativity and integrality:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$

##### Complete Numerical Formulation

Let $I = \{F1, F2, \ldots, F15\}$ and $J = \{C1, C2, \ldots, C15\}$.

Parameters (all coefficients from source):

- $f_{F1} = 11250$, $cap_{F1} = 101$, $c_{F1,C1} = 7.8$, $c_{F1,C2} = 7.6$, ..., $c_{F1,C15} = 7.9$
- $f_{F2} = 13480$, $cap_{F2} = 124$, $c_{F2,C1} = 5.3$, ..., $c_{F2,C15} = 6.3$
- $f_{F3} = 14870$, $cap_{F3} = 139$, $c_{F3,C1} = 7.2$, ..., $c_{F3,C15} = 8.4$
- $f_{F4} = 10290$, $cap_{F4} = 86$, $c_{F4,C1} = 7$, ..., $c_{F4,C15} = 7.6$
- $f_{F5} = 16740$, $cap_{F5} = 157$, $c_{F5,C1} = 3.5$, ..., $c_{F5,C15} = 3.9$
- $f_{F6} = 13960$, $cap_{F6} = 133$, $c_{F6,C1} = 8.2$, ..., $c_{F6,C15} = 9$
- $f_{F7} = 12680$, $cap_{F7} = 118$, $c_{F7,C1} = 6.9$, ..., $c_{F7,C15} = 7.5$
- $f_{F8} = 17890$, $cap_{F8} = 162$, $c_{F8,C1} = 6.9$, ..., $c_{F8,C15} = 8$
- $f_{F9} = 10950$, $cap_{F9} = 92$, $c_{F9,C1} = 3.5$, ..., $c_{F9,C15} = 4.5$
- $f_{F10} = 15320$, $cap_{F10} = 144$, $c_{F10,C1} = 5.2$, ..., $c_{F10,C15} = 6.2$
- $f_{F11} = 11830$, $cap_{F11} = 107$, $c_{F11,C1} = 5.2$, ..., $c_{F11,C15} = 5.4$
- $f_{F12} = 14110$, $cap_{F12} = 129$, $c_{F12,C1} = 7.8$, ..., $c_{F12,C15} = 8.7$
- $f_{F13} = 15970$, $cap_{F13} = 151$, $c_{F13,C1} = 6.7$, ..., $c_{F13,C15} = 7.2$
- $f_{F14} = 13140$, $cap_{F14} = 113$, $c_{F14,C1} = 7.5$, ..., $c_{F14,C15} = 8$
- $f_{F15} = 10580$, $cap_{F15} = 85$, $c_{F15,C1} = 5.1$, ..., $c_{F15,C15} = 6$

Customer demands:

- $d_{C1} = 83$, $d_{C2} = 76$, $d_{C3} = 91$, $d_{C4} = 68$, $d_{C5} = 104$, $d_{C6} = 97$, $d_{C7} = 88$, $d_{C8} = 73$, $d_{C9} = 109$, $d_{C10} = 95$, $d_{C11} = 82$, $d_{C12} = 67$, $d_{C13} = 113$, $d_{C14} = 79$, $d_{C15} = 92$

Variables:

- $y_i \in \{0,1\}$ for $i \in I$
- $x_{ij} \geq 0$ for $i \in I, j \in J$

Objective:

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

- For all $j \in J$:
  $$
  \sum_{i \in I} x_{ij} \geq d_j
  $$
- For all $i \in I$:
  $$
  \sum_{j \in J} x_{ij} \leq cap_i \cdot y_i
  $$
- $x_{ij} \geq 0$ for all $i \in I, j \in J$
- $y_i \in \{0,1\}$ for all $i \in I$