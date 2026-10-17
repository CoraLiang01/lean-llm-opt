##### Decision Variables

$x_{ij} \geq 0$: Number of units of product $j \in J$ placed on shelf $i \in I$ (continuous or integer, as appropriate).

##### Parameters

- Shelves $I = \{1,2,3,4,5,6,7,8,9,10\}$
- Products $J = \{1,2,3,\ldots,20\}$

- Shelf capacities:
  - $C_1 = 750$
  - $C_2 = 820$
  - $C_3 = 570$
  - $C_4 = 800$
  - $C_5 = 550$
  - $C_6 = 900$
  - $C_7 = 650$
  - $C_8 = 800$
  - $C_9 = 850$
  - $C_{10} = 900$

- Product values and weights:

| $j$ | Value $v_j$ | Weight $w_j$ |
|-----|-------------|--------------|
| 1   | 55          | 10           |
| 2   | 75          | 20           |
| 3   | 65          | 5            |
| 4   | 60          | 15           |
| 5   | 80          | 25           |
| 6   | 90          | 35           |
| 7   | 40          | 45           |
| 8   | 100         | 55           |
| 9   | 55          | 65           |
| 10  | 75          | 20           |
| 11  | 110         | 18           |
| 12  | 50          | 28           |
| 13  | 60          | 8            |
| 14  | 120         | 28           |
| 15  | 70          | 25           |
| 16  | 110         | 40           |
| 17  | 50          | 55           |
| 18  | 60          | 70           |
| 19  | 120         | 85           |
| 20  | 100         | 100          |

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Shelf capacity constraints:** For each shelf $i \in I$,
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i
   \]
2. **Nonnegativity:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

##### Retrieved Information

- Shelves: $I = \{1,2,3,4,5,6,7,8,9,10\}$
- Shelf capacities: $C_1 = 750$, $C_2 = 820$, $C_3 = 570$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 800$, $C_9 = 850$, $C_{10} = 900$
- Products: $J = \{1,2,\ldots,20\}$
- Product values and weights as listed above.