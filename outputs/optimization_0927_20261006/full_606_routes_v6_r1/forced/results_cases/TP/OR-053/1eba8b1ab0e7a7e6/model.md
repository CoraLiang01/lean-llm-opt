##### Sets

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves (ShelfID from capacity.csv).

Let $J = \{1,2,\ldots,20\}$ be the set of products (ProductName from products.csv).

##### Parameters

For each shelf $i \in I$:

- $C_i$ = capacity of shelf $i$.

From capacity.csv:
- $C_1 = 500$
- $C_2 = 700$
- $C_3 = 600$
- $C_4 = 800$
- $C_5 = 550$
- $C_6 = 900$
- $C_7 = 650$
- $C_8 = 750$
- $C_9 = 820$
- $C_{10} = 570$

For each product $j \in J$:

- $v_j$ = value of product $j$
- $w_j$ = weight of product $j$

From products.csv:

| $j$ | $v_j$ | $w_j$ |
|-----|-------|-------|
| 1   | 50    | 10    |
| 2   | 70    | 20    |
| 3   | 30    | 5     |
| 4   | 60    | 15    |
| 5   | 80    | 25    |
| 6   | 90    | 30    |
| 7   | 40    | 12    |
| 8   | 100   | 35    |
| 9   | 55    | 10    |
| 10  | 75    | 20    |
| 11  | 65    | 18    |
| 12  | 95    | 28    |
| 13  | 45    | 8     |
| 14  | 85    | 22    |
| 15  | 70    | 25    |
| 16  | 110   | 40    |
| 17  | 50    | 14    |
| 18  | 60    | 16    |
| 19  | 120   | 50    |
| 20  | 100   | 30    |

##### Decision Variables

For each shelf $i \in I$ and product $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$.

##### Objective

Maximize total value of products placed on all shelves:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

1. Shelf capacity constraints (for each $i \in I$):
   $$
   \sum_{j \in J} w_j x_{ij} \leq C_i
   $$

2. Nonnegativity and integrality:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   $$

##### Complete Model

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
$$

Where the parameters $C_i$, $v_j$, $w_j$ are as listed above.

###### Retrieved Information

- Shelves and capacities (capacity.csv, in source order):

  1: 500, 2: 700, 3: 600, 4: 800, 5: 550, 6: 900, 7: 650, 8: 750, 9: 820, 10: 570

- Products, values, and weights (products.csv, in source order):

  1: 50/10, 2: 70/20, 3: 30/5, 4: 60/15, 5: 80/25, 6: 90/30, 7: 40/12, 8: 100/35, 9: 55/10, 10: 75/20, 11: 65/18, 12: 95/28, 13: 45/8, 14: 85/22, 15: 70/25, 16: 110/40, 17: 50/14, 18: 60/16, 19: 120/50, 20: 100/30