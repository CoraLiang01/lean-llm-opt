Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i$ indexes shelves and $j$ indexes products. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Shelf capacities ($C_i$):  
  $C_1 = 5.0$,  
  $C_2 = 7.0$,  
  $C_3 = 6.0$,  
  $C_4 = 8.0$,  
  $C_5 = 5.5$,  
  $C_6 = 9.0$,  
  $C_7 = 6.5$,  
  $C_8 = 7.5$,  
  $C_9 = 8.2$,  
  $C_{10} = 5.7$

- Products ($j$) and their values ($v_j$) and weights ($w_j$):

| ProductName            | $v_j$ | $w_j$  |
|------------------------|-------|--------|
| Smartphone             |  200  |  1.0   |
| Laptop                 | 1500  |  5.0   |
| Headphones             |  100  |  0.5   |
| Camera                 |  800  |  2.0   |
| Smartwatch             |  250  |  0.3   |
| Tablet                 |  600  |  1.5   |
| Bluetooth Speaker      |  150  |  1.0   |
| Keyboard               |   80  |  0.8   |
| Mouse                  |   50  |  0.2   |
| Monitor                |  300  |  3.0   |
| Printer                |  400  |  4.0   |
| External Hard Drive    |  120  |  0.5   |
| Router                 |   60  |  0.3   |
| Power Bank             |   40  |  0.4   |
| Memory Card            |   30  | 0.05   |
| USB Flash Drive        |   25  | 0.02   |
| Smart Home Hub         |  100  |  0.6   |
| Gaming Console         |  500  |  4.0   |
| Fitness Tracker        |   90  |  0.2   |
| E-Reader               |  180  |  0.5   |

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

**Subject to:**

- Shelf capacity constraints (for each shelf $i$):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,20\}
\]

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight of product $j$ (see table above)
- $C_i$: Capacity of shelf $i$ (see above)

**All identifiers and coefficients are as retrieved from the source data.**