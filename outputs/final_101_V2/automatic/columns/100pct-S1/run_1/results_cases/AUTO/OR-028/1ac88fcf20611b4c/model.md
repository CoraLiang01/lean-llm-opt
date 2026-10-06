##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i$ to store $j$ (continuous), for all warehouses $i \in I$ and stores $j \in J$.
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (Warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (Stores)

- Warehouse opening costs $f_i$ and capacities $K_i$:

| $i$ | $f_i$ (Opening Cost) | $K_i$ (Capacity) |
|-----|----------------------|------------------|
| 1   | 3000                 | 180              |
| 2   | 3200                 | 160              |
| 3   | 3100                 | 200              |
| 4   | 2800                 | 150              |
| 5   | 3500                 | 170              |
| 6   | 2700                 | 190              |
| 7   | 2900                 | 160              |
| 8   | 3050                 | 175              |
| 9   | 3100                 | 170              |
| 10  | 2200                 | 180              |
| 11  | 2890                 | 190              |

- Store demands $d_j$:

| $j$ | $d_j$ (Demand) |
|-----|---------------|
| 1   | 30            |
| 2   | 40            |
| 3   | 20            |
| 4   | 35            |
| 5   | 20            |
| 6   | 25            |
| 7   | 45            |
| 8   | 38            |
| 9   | 32            |
| 10  | 41            |
| 11  | 44            |

- Transportation costs $c_{ij}$ (from warehouse $i$ to store $j$):

| $i$  | $j=1$ | $j=2$ | $j=3$ | $j=4$ | $j=5$ | $j=6$ | $j=7$ | $j=8$ | $j=9$ | $j=10$ | $j=11$ |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1    | 12    | 17    | 13    | 18    | 10    | 15    | 14    | 19    | 17    | 14     | 15     |
| 2    | 11    | 19    | 14    | 16    | 13    | 12    | 13    | 16    | 18    | 13     | 13     |
| 3    | 14    | 15    | 12    | 17    | 12    | 14    | 15    | 18    | 12    | 15     | 16     |
| 4    | 15    | 20    | 14    | 13    | 19    | 16    | 17    | 20    | 14    | 17     | 17     |
| 5    | 17    | 18    | 16    | 18    | 15    | 13    | 12    | 17    | 16    | 16     | 11     |
| 6    | 13    | 14    | 15    | 17    | 11    | 17    | 13    | 19    | 15    | 18     | 13     |
| 7    | 12    | 17    | 11    | 14    | 12    | 16    | 14    | 16    | 14    | 14     | 14     |
| 8    | 16    | 15    | 14    | 19    | 14    | 16    | 15    | 18    | 17    | 19     | 15     |
| 9    | 16    | 13    | 16    | 16    | 12    | 14    | 12    | 15    | 21    | 15     | 19     |
| 10   | 14    | 15    | 18    | 13    | 15    | 18    | 16    | 15    | 15    | 17     | 21     |
| 11   | 15    | 16    | 17    | 18    | 17    | 19    | 14    | 18    | 18    | 19     | 13     |

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
   \sum_{j=1}^{11} x_{ij} \leq K_i y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Summary of Sets and Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (Warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (Stores)
- $f_i$, $K_i$ as above for each $i$
- $d_j$ as above for each $j$
- $c_{ij}$ as above for each $i,j$

All data, indices, and coefficients are preserved as in the original CSVs.