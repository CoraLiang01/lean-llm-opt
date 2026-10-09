##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity (only if opened):**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{Cap}_i \cdot y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (stores)

###### Warehouse Opening Costs and Capacities

| Warehouse (i) | Opening Cost $f_i$ | Capacity (units) $\text{Cap}_i$ |
|:-------------:|:------------------:|:-------------------------------:|
| 1             | 3000               | 180                             |
| 2             | 3200               | 160                             |
| 3             | 3100               | 200                             |
| 4             | 2800               | 150                             |
| 5             | 3500               | 170                             |
| 6             | 2700               | 190                             |
| 7             | 2900               | 160                             |
| 8             | 3050               | 175                             |
| 9             | 3100               | 170                             |
| 10            | 2200               | 180                             |
| 11            | 2890               | 190                             |

###### Store Demands

| Store (j) | Demand $d_j$ |
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

###### Transportation Cost Matrix $c_{ij}$

The cost $c_{ij}$ is the per-unit transportation cost from warehouse $i$ to store $j$.

|        | Store 1 | Store 2 | Store 3 | Store 4 | Store 5 | Store 6 | Store 7 | Store 8 | Store 9 | Store 10 | Store 11 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|----------|
| **W1**  | 12      | 11      | 14      | 15      | 17      | 13      | 12      | 16      | 16      | 14       | 15       |
| **W2**  | 17      | 19      | 15      | 20      | 18      | 14      | 17      | 15      | 13      | 15       | 16       |
| **W3**  | 13      | 14      | 12      | 14      | 16      | 15      | 11      | 14      | 16      | 18       | 17       |
| **W4**  | 18      | 16      | 17      | 13      | 18      | 17      | 14      | 19      | 16      | 13       | 18       |
| **W5**  | 10      | 13      | 12      | 19      | 15      | 11      | 12      | 14      | 12      | 15       | 17       |
| **W6**  | 15      | 12      | 14      | 16      | 13      | 17      | 16      | 16      | 14      | 18       | 19       |
| **W7**  | 14      | 13      | 15      | 17      | 12      | 13      | 14      | 15      | 12      | 16       | 14       |
| **W8**  | 19      | 16      | 18      | 20      | 17      | 19      | 16      | 18      | 15      | 15       | 18       |
| **W9**  | 17      | 18      | 12      | 14      | 16      | 15      | 14      | 17      | 21      | 15       | 19       |
| **W10** | 14      | 13      | 15      | 17      | 16      | 18      | 14      | 19      | 15      | 17       | 19       |
| **W11** | 15      | 13      | 16      | 17      | 11      | 13      | 14      | 15      | 19      | 21       | 13       |

##### Model Summary

- Decide which warehouses to open ($y_i$).
- Decide how much to ship from each open warehouse to each store ($x_{ij}$).
- Meet all store demands exactly.
- Do not exceed the capacity of any open warehouse.
- Minimize total cost (opening + transportation).

All parameters and indices are as above, with warehouse $i$ corresponding to $W1$ through $W11$ and store $j$ corresponding to $1$ through $11$.