##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$, for each shelf $i \in S$ and product $j \in P$.

##### Parameters

- Shelves $S = \{1,2,3,4,5,6,7,8,9,10\}$
- Products $P = \{1,2,3,\ldots,20\}$

- Shelf capacities:
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

- Product values and weights:

| Product $j$ | $v_j$ (Value) | $w_j$ (Weight) |
|:-----------:|:-------------:|:--------------:|
| 1           | 50            | 10             |
| 2           | 70            | 20             |
| 3           | 30            | 5              |
| 4           | 60            | 15             |
| 5           | 80            | 25             |
| 6           | 90            | 30             |
| 7           | 40            | 12             |
| 8           | 100           | 35             |
| 9           | 55            | 10             |
| 10          | 75            | 20             |
| 11          | 65            | 18             |
| 12          | 95            | 28             |
| 13          | 45            | 8              |
| 14          | 85            | 22             |
| 15          | 70            | 25             |
| 16          | 110           | 40             |
| 17          | 50            | 14             |
| 18          | 60            | 16             |
| 19          | 120           | 50             |
| 20          | 100           | 30             |

##### Objective Function

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. **Shelf capacity constraints:** For each shelf $i \in S$,
   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i
   \]

2. **Nonnegativity and integrality:** For all $i \in S$, $j \in P$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

##### Complete Parameter Listing

- $S = \{1,2,3,4,5,6,7,8,9,10\}$
- $P = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$
- Shelf capacities:
  - $C_1 = 500$, $C_2 = 700$, $C_3 = 600$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 750$, $C_9 = 820$, $C_{10} = 570$
- Product values and weights:
  - $v = [50, 70, 30, 60, 80, 90, 40, 100, 55, 75, 65, 95, 45, 85, 70, 110, 50, 60, 120, 100]$
  - $w = [10, 20, 5, 15, 25, 30, 12, 35, 10, 20, 18, 28, 8, 22, 25, 40, 14, 16, 50, 30]$