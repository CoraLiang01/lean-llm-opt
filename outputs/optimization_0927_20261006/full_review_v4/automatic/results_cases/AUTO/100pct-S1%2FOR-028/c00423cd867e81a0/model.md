##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (stores)

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

Let $c_{ij}$ be the cost from warehouse $i$ to store $j$, where $i$ and $j$ both run from 1 to 11. The matrix below gives $c_{ij}$, with rows as $i$ (warehouses) and columns as $j$ (stores):

| $c_{ij}$ | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|----------|----|----|----|----|----|----|----|----|----|----|----|
| 1        | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| 2        | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| 3        | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| 4        | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| 5        | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
| 6        | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| 7        | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| 8        | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| 9        | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| 10       | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| 11       | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

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

##### All parameters (as retrieved):

- Warehouses $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Stores $J = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Opening costs $f_i$:
  - $f_1=3000$, $f_2=3200$, $f_3=3100$, $f_4=2800$, $f_5=3500$, $f_6=2700$, $f_7=2900$, $f_8=3050$, $f_9=3100$, $f_{10}=2200$, $f_{11}=2890$
- Capacities $u_i$:
  - $u_1=180$, $u_2=160$, $u_3=200$, $u_4=150$, $u_5=170$, $u_6=190$, $u_7=160$, $u_8=175$, $u_9=170$, $u_{10}=180$, $u_{11}=190$
- Demands $d_j$:
  - $d_1=30$, $d_2=40$, $d_3=20$, $d_4=35$, $d_5=20$, $d_6=25$, $d_7=45$, $d_8=38$, $d_9=32$, $d_{10}=41$, $d_{11}=44$
- Transportation costs $c_{ij}$ as in the matrix above.

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} + \sum_{i=1}^{11} f_i y_i \\
\text{s.t.}\quad
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j=1,\ldots,11 \\
& \sum_{j=1}^{11} x_{ij} \leq u_i y_i \quad \forall i=1,\ldots,11 \\
& x_{ij} \geq 0 \quad \forall i,j \\
& y_i \in \{0,1\} \quad \forall i
\end{align*}
\]

where all parameters are as listed above.