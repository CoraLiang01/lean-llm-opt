##### Sets

- $I = \{1,2,\ldots,11\}$: set of potential warehouses  
- $J = \{1,2,\ldots,11\}$: set of stores

##### Parameters

- $f_i$: opening cost of warehouse $i$  
  $f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]$  
  (for $i=1$ to $11$ in order)

- $K_i$: capacity of warehouse $i$  
  $K = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]$

- $d_j$: demand of store $j$  
  $d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]$

- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$  
  (matrix below: row $i$ = warehouse $i$, column $j$ = store $j$)

\[
C = \begin{bmatrix}
12 & 11 & 14 & 15 & 17 & 13 & 12 & 16 & 16 & 14 & 15 \\
17 & 19 & 15 & 20 & 18 & 14 & 17 & 15 & 13 & 15 & 16 \\
13 & 14 & 12 & 14 & 16 & 15 & 11 & 14 & 16 & 18 & 17 \\
18 & 16 & 17 & 13 & 18 & 17 & 14 & 19 & 16 & 13 & 18 \\
10 & 13 & 12 & 19 & 15 & 11 & 12 & 14 & 12 & 15 & 17 \\
15 & 12 & 14 & 16 & 13 & 17 & 16 & 16 & 14 & 18 & 19 \\
14 & 13 & 15 & 17 & 12 & 13 & 14 & 15 & 12 & 16 & 14 \\
19 & 16 & 18 & 20 & 17 & 19 & 16 & 18 & 15 & 15 & 18 \\
17 & 18 & 12 & 14 & 16 & 15 & 14 & 17 & 21 & 15 & 18 \\
14 & 13 & 15 & 17 & 16 & 18 & 14 & 19 & 15 & 17 & 19 \\
15 & 13 & 16 & 17 & 11 & 13 & 14 & 15 & 19 & 21 & 13 \\
\end{bmatrix}
\]

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

##### Parameter Values

- $f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]$
- $K = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]$
- $d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]$
- $C$ as above (rows: $i=1$ to $11$, columns: $j=1$ to $11$)

##### Indices

- $i = 1,2,\ldots,11$ (warehouses)
- $j = 1,2,\ldots,11$ (stores)