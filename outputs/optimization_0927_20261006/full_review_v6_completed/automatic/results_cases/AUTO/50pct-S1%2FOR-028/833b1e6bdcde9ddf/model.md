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

Let $c_{ij}$ be the cost per unit shipped from warehouse $i$ to store $j$, where $i$ corresponds to $W1$ through $W11$ and $j$ corresponds to the row index (store $j$).

| $c_{ij}$ | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|----------|----|----|----|----|----|----|----|----|----|-----|-----|
| Store 1  | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15  |
| Store 2  | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16  |
| Store 3  | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17  |
| Store 4  | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18  |
| Store 5  | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17  |
| Store 6  | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19  |
| Store 7  | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14  |
| Store 8  | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18  |
| Store 9  | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18  |
| Store 10 | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19  |
| Store 11 | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13  |

##### Objective Function

\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j$,
   \[
   \sum_{i=1}^{11} x_{ij} = d_j \qquad \forall j = 1,\ldots,11
   \]

2. **Warehouse capacity:**  
   For each warehouse $i$,
   \[
   \sum_{j=1}^{11} x_{ij} \leq u_i y_i \qquad \forall i = 1,\ldots,11
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \qquad \forall i,j
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i
   \]

##### All Parameters (as retrieved)

- $f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]$
- $u = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]$
- $d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]$
- $c_{ij}$ as in the table above, with $i$ indexing warehouses $1$ to $11$ and $j$ indexing stores $1$ to $11$.

##### Sets

- Warehouses $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Stores $J = \{1,2,3,4,5,6,7,8,9,10,11\}$

---

**This model determines which warehouses to open and how much to ship from each open warehouse to each store, minimizing total cost while meeting all demands and not exceeding warehouse capacities.**