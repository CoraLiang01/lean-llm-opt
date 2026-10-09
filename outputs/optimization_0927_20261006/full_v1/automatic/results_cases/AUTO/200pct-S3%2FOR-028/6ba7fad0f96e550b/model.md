##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (stores)

- Warehouse opening costs $f_i$ and capacities $K_i$:

| $i$ | $f_i$ | $K_i$ |
|-----|-------|-------|
| 1   | 3000  | 180   |
| 2   | 3200  | 160   |
| 3   | 3100  | 200   |
| 4   | 2800  | 150   |
| 5   | 3500  | 170   |
| 6   | 2700  | 190   |
| 7   | 2900  | 160   |
| 8   | 3050  | 175   |
| 9   | 3100  | 170   |
| 10  | 2200  | 180   |
| 11  | 2890  | 190   |

- Store demands $d_j$:

| $j$ | $d_j$ |
|-----|-------|
| 1   | 30    |
| 2   | 40    |
| 3   | 20    |
| 4   | 35    |
| 5   | 20    |
| 6   | 25    |
| 7   | 45    |
| 8   | 38    |
| 9   | 32    |
| 10  | 41    |
| 11  | 44    |

- Transportation costs $c_{ij}$ (rows: stores $j=1,\ldots,11$; columns: warehouses $i=1,\ldots,11$):

| $j \backslash i$ | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|------------------|----|----|----|----|----|----|----|----|----|----|----|
| 1                | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| 2                | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| 3                | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| 4                | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| 5                | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 11 |
| 6                | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| 7                | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| 8                | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| 9                | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 19 |
| 10               | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| 11               | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse capacity (only if opened):**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

---

###### Retrieved Information

- Warehouses $I = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Stores $J = \{1,2,3,4,5,6,7,8,9,10,11\}$
- Opening costs $f_i$ and capacities $K_i$ as above
- Demands $d_j$ as above
- Transportation cost matrix $c_{ij}$ as above (row $j$, column $i$)