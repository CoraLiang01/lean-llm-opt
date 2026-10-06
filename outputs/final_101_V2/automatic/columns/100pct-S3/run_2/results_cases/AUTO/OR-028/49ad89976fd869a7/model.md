##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$ (binary).

##### Parameters

- $I = \{1,2,\ldots,11\}$: Set of warehouses (corresponding to W1–W11).
- $J = \{1,2,\ldots,11\}$: Set of stores.
- $f_i$: Opening cost for warehouse $i$.
- $cap_i$: Capacity of warehouse $i$.
- $d_j$: Demand of store $j$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$.

###### Warehouse Opening Costs and Capacities

| $i$ | $f_i$ | $cap_i$ |
|-----|-------|---------|
| 1   | 3000  | 180     |
| 2   | 3200  | 160     |
| 3   | 3100  | 200     |
| 4   | 2800  | 150     |
| 5   | 3500  | 170     |
| 6   | 2700  | 190     |
| 7   | 2900  | 160     |
| 8   | 3050  | 175     |
| 9   | 3100  | 170     |
| 10  | 2200  | 180     |
| 11  | 2890  | 190     |

###### Store Demands

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

###### Transportation Costs $c_{ij}$

| $i$ (W) | $j=1$ | $j=2$ | $j=3$ | $j=4$ | $j=5$ | $j=6$ | $j=7$ | $j=8$ | $j=9$ | $j=10$ | $j=11$ |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1       | 12    | 11    | 14    | 15    | 17    | 13    | 12    | 16    | 16    | 14     | 15     |
| 2       | 17    | 19    | 15    | 20    | 18    | 14    | 17    | 15    | 13    | 15     | 16     |
| 3       | 13    | 14    | 12    | 14    | 16    | 15    | 11    | 14    | 16    | 18     | 17     |
| 4       | 18    | 16    | 17    | 13    | 18    | 17    | 14    | 19    | 16    | 13     | 18     |
| 5       | 10    | 13    | 12    | 19    | 15    | 11    | 12    | 14    | 12    | 15     | 17     |
| 6       | 15    | 12    | 14    | 16    | 13    | 17    | 16    | 16    | 14    | 18     | 19     |
| 7       | 14    | 13    | 15    | 17    | 12    | 13    | 14    | 15    | 12    | 16     | 14     |
| 8       | 19    | 16    | 18    | 20    | 17    | 19    | 16    | 18    | 15    | 15     | 18     |
| 9       | 17    | 18    | 12    | 14    | 16    | 15    | 14    | 17    | 21    | 15     | 18     |
| 10      | 14    | 13    | 15    | 17    | 16    | 18    | 14    | 19    | 15    | 17     | 19     |
| 11      | 15    | 13    | 16    | 17    | 11    | 13    | 14    | 15    | 19    | 21     | 13     |

##### Mathematical Model

**Objective:**
\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i=1}^{11} x_{ij} = d_j \qquad \forall j = 1,\ldots,11
   \]

2. **Warehouse capacity (cannot exceed capacity if open, cannot ship if closed):**
   \[
   \sum_{j=1}^{11} x_{ij} \leq cap_i \cdot y_i \qquad \forall i = 1,\ldots,11
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \qquad \forall i, j
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i
   \]

**Where:**

- $f_i$, $cap_i$, $d_j$, and $c_{ij}$ are as given in the tables above.
- $x_{ij}$ and $y_i$ are the decision variables.

**All parameters and indices are preserved as in the original data.**