##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (stores)

- Warehouse opening costs and capacities:
  - $f_1 = 3000$, $u_1 = 180$
  - $f_2 = 3200$, $u_2 = 160$
  - $f_3 = 3100$, $u_3 = 200$
  - $f_4 = 2800$, $u_4 = 150$
  - $f_5 = 3500$, $u_5 = 170$
  - $f_6 = 2700$, $u_6 = 190$
  - $f_7 = 2900$, $u_7 = 160$
  - $f_8 = 3050$, $u_8 = 175$
  - $f_9 = 3100$, $u_9 = 170$
  - $f_{10} = 2200$, $u_{10} = 180$
  - $f_{11} = 2890$, $u_{11} = 190$

- Store demands:
  - $d_1 = 30$
  - $d_2 = 40$
  - $d_3 = 20$
  - $d_4 = 35$
  - $d_5 = 20$
  - $d_6 = 25$
  - $d_7 = 45$
  - $d_8 = 38$
  - $d_9 = 32$
  - $d_{10} = 41$
  - $d_{11} = 44$

- Transportation costs $c_{ij}$ (from warehouse $i$ to store $j$):

|         | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|---------|----|----|----|----|----|----|----|----|----|----|----|
| **1**   | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| **2**   | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| **3**   | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| **4**   | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| **5**   | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
| **6**   | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| **7**   | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| **8**   | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| **9**   | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| **10**  | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| **11**  | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
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

---

###### Retrieved Information

- Warehouses $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ with opening costs and capacities:
  - $f_1 = 3000$, $u_1 = 180$
  - $f_2 = 3200$, $u_2 = 160$
  - $f_3 = 3100$, $u_3 = 200$
  - $f_4 = 2800$, $u_4 = 150$
  - $f_5 = 3500$, $u_5 = 170$
  - $f_6 = 2700$, $u_6 = 190$
  - $f_7 = 2900$, $u_7 = 160$
  - $f_8 = 3050$, $u_8 = 175$
  - $f_9 = 3100$, $u_9 = 170$
  - $f_{10} = 2200$, $u_{10} = 180$
  - $f_{11} = 2890$, $u_{11} = 190$

- Stores $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ with demands:
  - $d_1 = 30$, $d_2 = 40$, $d_3 = 20$, $d_4 = 35$, $d_5 = 20$, $d_6 = 25$, $d_7 = 45$, $d_8 = 38$, $d_9 = 32$, $d_{10} = 41$, $d_{11} = 44$

- Transportation cost matrix $c_{ij}$ as above, with $i$ as warehouse and $j$ as store.