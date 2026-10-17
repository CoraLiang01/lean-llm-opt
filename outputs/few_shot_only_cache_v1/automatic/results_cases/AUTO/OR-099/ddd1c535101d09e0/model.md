##### Sets

- Warehouses: $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Stores: $J = \{1,2,3,4,5,6,7,8,9,10,11\}$

##### Parameters

- Opening cost for warehouse $i$: $f_i$

  $f_1 = 3000$, $f_2 = 3200$, $f_3 = 3100$, $f_4 = 2800$, $f_5 = 3500$, $f_6 = 2700$, $f_7 = 2900$, $f_8 = 3050$, $f_9 = 3100$, $f_{10} = 2200$, $f_{11} = 2890$

- Capacity of warehouse $i$: $u_i$

  $u_1 = 180$, $u_2 = 160$, $u_3 = 200$, $u_4 = 150$, $u_5 = 170$, $u_6 = 190$, $u_7 = 160$, $u_8 = 175$, $u_9 = 170$, $u_{10} = 180$, $u_{11} = 190$

- Demand of store $j$: $d_j$

  $d_1 = 30$, $d_2 = 40$, $d_3 = 20$, $d_4 = 35$, $d_5 = 20$, $d_6 = 25$, $d_7 = 45$, $d_8 = 38$, $d_9 = 32$, $d_{10} = 41$, $d_{11} = 44$

- Transportation cost per unit from warehouse $i$ to store $j$: $c_{ij}$

  The transportation cost matrix $C = [c_{ij}]$ is as follows (rows: warehouses $i=1$ to $11$, columns: stores $j=1$ to $11$):

  |        | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
  |--------|----|----|----|----|----|----|----|----|----|----|----|
  | **1**  | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
  | **2**  | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
  | **3**  | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
  | **4**  | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
  | **5**  | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
  | **6**  | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
  | **7**  | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
  | **8**  | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
  | **9**  | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
  | **10** | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
  | **11** | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be fully met:
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]

2. **Warehouse capacity:** Shipments from each warehouse cannot exceed its capacity if opened:
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameters (full listing)

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$
- $f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]$
- $u = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]$
- $d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]$
- $C = [c_{ij}]$ as given in the table above.

##### Model Summary

- Decide which warehouses to open ($y_i$) and how much to ship from each warehouse to each store ($x_{ij}$).
- Minimize total cost (opening + transportation).
- Meet all store demands.
- Do not exceed warehouse capacities if opened.
- All variables and parameters are as listed above.