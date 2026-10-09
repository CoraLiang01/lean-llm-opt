##### Sets

- $I = \{1,2,\ldots,11\}$: set of potential warehouses  
- $J = \{1,2,\ldots,11\}$: set of stores

##### Parameters

- $f_i$: opening cost of warehouse $i$  
- $K_i$: capacity of warehouse $i$ (units)  
- $d_j$: demand of store $j$ (units)  
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

**From PotentialWarehouses_Costs.csv:**

| $i$ | $f_i$ | $K_i$ |
|---|-------|-------|
| 1 | 3000  | 180   |
| 2 | 3200  | 160   |
| 3 | 3100  | 200   |
| 4 | 2800  | 150   |
| 5 | 3500  | 170   |
| 6 | 2700  | 190   |
| 7 | 2900  | 160   |
| 8 | 3050  | 175   |
| 9 | 3100  | 170   |
|10 | 2200  | 180   |
|11 | 2890  | 190   |

**From Stores_Demands.csv:**

| $j$ | $d_j$ |
|---|------|
| 1 | 30   |
| 2 | 40   |
| 3 | 20   |
| 4 | 35   |
| 5 | 20   |
| 6 | 25   |
| 7 | 45   |
| 8 | 38   |
| 9 | 32   |
|10 | 41   |
|11 | 44   |

**From TransportationCost.csv:**  
$c_{ij}$ is the cost per unit from warehouse $i$ to store $j$ (rows: $i$ as $W1$–$W11$, columns: $j$ as $W1$–$W11$):

| $c_{ij}$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|----------|---|---|---|---|---|---|---|---|---|----|----|
| 1        |12 |11 |14 |15 |17 |13 |12 |16 |16 |14  |15  |
| 2        |17 |19 |15 |20 |18 |14 |17 |15 |13 |15  |16  |
| 3        |13 |14 |12 |14 |16 |15 |11 |14 |16 |18  |17  |
| 4        |18 |16 |17 |13 |18 |17 |14 |19 |16 |13  |18  |
| 5        |10 |13 |12 |19 |15 |11 |12 |14 |12 |15  |17  |
| 6        |15 |12 |14 |16 |13 |17 |16 |16 |14 |18  |19  |
| 7        |14 |13 |15 |17 |12 |13 |14 |15 |12 |16  |14  |
| 8        |19 |16 |18 |20 |17 |19 |16 |18 |15 |15  |18  |
| 9        |17 |18 |12 |14 |16 |15 |14 |17 |21 |15  |18  |
|10        |14 |13 |15 |17 |16 |18 |14 |19 |15 |17  |19  |
|11        |15 |13 |16 |17 |11 |13 |14 |15 |19 |21  |13  |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for $i \in I$, $j \in J$

##### Objective Function

\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
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
   \sum_{j=1}^{11} x_{ij} \leq K_i y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Parameter Listing

- $I = J = \{1,2,\ldots,11\}$
- $f = [3000,\,3200,\,3100,\,2800,\,3500,\,2700,\,2900,\,3050,\,3100,\,2200,\,2890]$
- $K = [180,\,160,\,200,\,150,\,170,\,190,\,160,\,175,\,170,\,180,\,190]$
- $d = [30,\,40,\,20,\,35,\,20,\,25,\,45,\,38,\,32,\,41,\,44]$
- $C = [$
  [12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15],
  [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16],
  [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17],
  [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18],
  [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17],
  [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19],
  [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14],
  [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18],
  [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18],
  [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19],
  [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]
$]$

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j=1,\ldots,11 \\
& \sum_{j=1}^{11} x_{ij} \leq K_i y_i \quad \forall i=1,\ldots,11 \\
& x_{ij} \geq 0 \quad \forall i=1,\ldots,11;\ j=1,\ldots,11 \\
& y_i \in \{0,1\} \quad \forall i=1,\ldots,11
\end{align*}
\]

where all parameters are as listed above.