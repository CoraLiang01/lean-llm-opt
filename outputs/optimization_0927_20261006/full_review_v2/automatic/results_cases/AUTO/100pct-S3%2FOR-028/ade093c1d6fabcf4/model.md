##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

\[
\min \sum_{i\in I} f_i y_i + \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity (only if opened):**  
   \[
   \sum_{j\in J} x_{ij} \leq \text{cap}_i \cdot y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

---

#### Parameters (retrieved from CSVs)

**Warehouses ($I$):**
| Warehouse $i$ | Opening Cost $f_i$ | Capacity $\text{cap}_i$ |
|:-------------:|:------------------:|:-----------------------:|
| 1             | 3000               | 180                     |
| 2             | 3200               | 160                     |
| 3             | 3100               | 200                     |
| 4             | 2800               | 150                     |
| 5             | 3500               | 170                     |
| 6             | 2700               | 190                     |
| 7             | 2900               | 160                     |
| 8             | 3050               | 175                     |
| 9             | 3100               | 170                     |
| 10            | 2200               | 180                     |
| 11            | 2890               | 190                     |

**Stores ($J$):**
| Store $j$ | Demand $d_j$ |
|:---------:|:------------:|
| 1         | 30           |
| 2         | 40           |
| 3         | 20           |
| 4         | 35           |
| 5         | 20           |
| 6         | 25           |
| 7         | 45           |
| 8         | 38           |
| 9         | 32           |
| 10        | 41           |
| 11        | 44           |

**Transportation Costs $c_{ij}$:**  
(warehouse rows $i$ = W1 to W11, store columns $j$ = 1 to 11)

| $c_{ij}$ | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|----------|----|----|----|----|----|----|----|----|----|----|----|
| W1       | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| W2       | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| W3       | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| W4       | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| W5       | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
| W6       | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| W7       | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| W8       | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| W9       | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| W10      | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| W11      | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

- $I = \{1,2,\ldots,11\}$ (warehouses)
- $J = \{1,2,\ldots,11\}$ (stores)
- $f_i$, $\text{cap}_i$ as above
- $d_j$ as above
- $c_{ij}$ as above (row $i$ = warehouse $i$, column $j$ = store $j$)

---

**Complete Mathematical Model:**

\[
\begin{align*}
\min \quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i=1}^{11} x_{ij} = d_j, \quad \forall j=1,\ldots,11 \\
& \sum_{j=1}^{11} x_{ij} \leq \text{cap}_i \cdot y_i, \quad \forall i=1,\ldots,11 \\
& x_{ij} \geq 0, \quad \forall i,j \\
& y_i \in \{0,1\}, \quad \forall i
\end{align*}
\]

with all parameters as listed above.