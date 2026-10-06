##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Customer demand satisfaction:
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Warehouse capacity (only if opened):
   \[
   \sum_{j\in J} x_{ij} \leq \text{cap}_i \cdot y_i, \quad \forall i \in I
   \]
3. Domains:
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters

- Warehouses $I = \{\text{W1}, \text{W2}, \text{W3}, \text{W4}, \text{W5}, \text{W6}, \text{W7}, \text{W8}, \text{W9}, \text{W10}\}$
- Customers $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}, \text{C13}, \text{C14}, \text{C15}, \text{C16}, \text{C17}, \text{C18}, \text{C19}, \text{C20}\}$

- Warehouse fixed costs $f_i$ and capacities $\text{cap}_i$:

| Warehouse ID | $f_i$ | $\text{cap}_i$ |
|--------------|-------|----------------|
| W1           | 2000  | 1000           |
| W2           | 2500  | 1500           |
| W3           | 1800  | 1200           |
| W4           | 3200  | 2000           |
| W5           | 1500  | 800            |
| W6           | 4000  | 2500           |
| W7           | 2800  | 1800           |
| W8           | 1950  | 1100           |
| W9           | 3500  | 2100           |
| W10          | 2200  | 1300           |

- Customer demands $d_j$:

| Customer ID | $d_j$ |
|-------------|-------|
| C1          | 800   |
| C2          | 600   |
| C3          | 500   |
| C4          | 700   |
| C5          | 450   |
| C6          | 950   |
| C7          | 350   |
| C8          | 850   |
| C9          | 400   |
| C10         | 750   |
| C11         | 900   |
| C12         | 550   |
| C13         | 650   |
| C14         | 820   |
| C15         | 480   |
| C16         | 920   |
| C17         | 320   |
| C18         | 780   |
| C19         | 520   |
| C20         | 680   |

- Transportation costs $c_{ij}$ (rows: warehouses $i$, columns: customers $j$):

|        | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|--------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W1     | 10 | 15 | 20 | 11 | 16 | 18 | 7  | 12 | 22 | 9   | 14  | 19  | 25  | 13  | 17  | 6   | 21  | 15  | 8   | 10  |
| W2     | 18 | 12 | 9  | 14 | 10 | 5  | 19 | 23 | 11 | 16  | 20  | 8   | 15  | 22  | 7   | 13  | 24  | 17  | 12  | 6   |
| W3     | 13 | 17 | 15 | 8  | 12 | 21 | 16 | 10 | 5  | 24  | 13  | 22  | 7   | 19  | 14  | 18  | 9   | 25  | 11  | 16  |
| W4     | 7  | 22 | 11 | 16 | 20 | 8  | 15 | 19 | 13 | 25  | 6   | 14  | 21  | 9   | 23  | 17  | 10  | 18  | 24  | 5   |
| W5     | 16 | 9  | 25 | 13 | 7  | 10 | 23 | 14 | 18 | 21  | 5   | 17  | 9   | 24  | 12  | 20  | 6   | 15  | 19  | 11  |
| W6     | 22 | 6  | 14 | 19 | 23 | 11 | 8  | 17 | 9  | 12  | 15  | 24  | 5   | 20  | 10  | 25  | 13  | 7   | 18  | 16  |
| W7     | 8  | 25 | 17 | 9  | 14 | 22 | 11 | 6  | 16 | 20  | 18  | 13  | 24  | 5   | 19  | 12  | 23  | 10  | 7   | 15  |
| W8     | 19 | 11 | 7  | 21 | 15 | 24 | 13 | 16 | 20 | 8   | 17  | 10  | 12  | 23  | 5   | 14  | 22  | 9   | 16  | 25  |
| W9     | 12 | 20 | 5  | 23 | 17 | 14 | 9  | 25 | 18 | 11  | 16  | 21  | 10  | 7   | 24  | 15  | 19  | 6   | 13  | 22  |
| W10    | 25 | 14 | 22 | 5  | 19 | 12 | 24 | 7  | 15 | 17  | 23  | 6   | 16  | 10  | 20  | 9   | 18  | 11  | 25  | 14  |