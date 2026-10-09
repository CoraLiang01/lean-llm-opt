##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- Warehouses $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Stores $J = \{1,2,3,4,5,6,7,8,9,10,11\}$

- Warehouse opening costs $f_i$ and capacities $u_i$:

| Warehouse $i$ | $f_i$ (Opening Cost) | $u_i$ (Capacity) |
|:-------------:|:-------------------:|:----------------:|
| 1             | 3000                | 180              |
| 2             | 3200                | 160              |
| 3             | 3100                | 200              |
| 4             | 2800                | 150              |
| 5             | 3500                | 170              |
| 6             | 2700                | 190              |
| 7             | 2900                | 160              |
| 8             | 3050                | 175              |
| 9             | 3100                | 170              |
| 10            | 2200                | 180              |
| 11            | 2890                | 190              |

- Store demands $d_j$:

| Store $j$ | $d_j$ (Demand) |
|:---------:|:--------------:|
| 1         | 30             |
| 2         | 40             |
| 3         | 20             |
| 4         | 35             |
| 5         | 20             |
| 6         | 25             |
| 7         | 45             |
| 8         | 38             |
| 9         | 32             |
| 10        | 41             |
| 11        | 44             |

- Transportation costs $c_{ij}$ (from warehouse $i$ to store $j$):

| $c_{ij}$ | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|----------|----|----|----|----|----|----|----|----|----|----|----|
| **1**    | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| **2**    | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| **3**    | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| **4**    | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| **5**    | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
| **6**    | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| **7**    | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| **8**    | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| **9**    | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| **10**   | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| **11**   | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

##### Objective Function

\[
\min \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} + \sum_{i=1}^{11} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i=1}^{11} x_{ij} = d_j
   \]
2. **Warehouse capacity:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j=1}^{11} x_{ij} \leq u_i y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (as retrieved)

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$
- $f_i = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]$
- $u_i = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]$
- $d_j = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]$
- $c_{ij}$ as in the table above, with $i$ as warehouse (row), $j$ as store (column).

##### Model Summary

- Decide which warehouses to open ($y_i$) and how much to ship from each warehouse to each store ($x_{ij}$) to minimize total cost.
- All store demands must be met.
- No warehouse can ship more than its capacity, and only if it is open.
- All variables and parameters are as above.