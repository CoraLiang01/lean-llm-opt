##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i$ to store $j$ (continuous), for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$.

##### Parameters

- $I = \{1,2,\ldots,11\}$: Set of potential warehouses.
- $J = \{1,2,\ldots,11\}$: Set of stores.
- $f_i$: Opening cost of warehouse $i$.
- $C_i$: Capacity of warehouse $i$.
- $d_j$: Demand of store $j$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$.

###### Warehouse opening costs and capacities (from PotentialWarehouses_Costs.csv):

| $i$ | $f_i$ | $C_i$ |
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

###### Store demands (from Stores_Demands.csv):

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

###### Transportation cost matrix $c_{ij}$ (from TransportationCost.csv):

| $i$ (Warehouse) | $j=1$ | $j=2$ | $j=3$ | $j=4$ | $j=5$ | $j=6$ | $j=7$ | $j=8$ | $j=9$ | $j=10$ | $j=11$ |
|-----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1 (W1)          | 12    | 11    | 14    | 15    | 17    | 13    | 12    | 16    | 16    | 14     | 15     |
| 2 (W2)          | 17    | 19    | 15    | 20    | 18    | 14    | 17    | 15    | 13    | 15     | 16     |
| 3 (W3)          | 13    | 14    | 12    | 14    | 16    | 15    | 11    | 14    | 16    | 18     | 17     |
| 4 (W4)          | 18    | 16    | 17    | 13    | 18    | 17    | 14    | 19    | 16    | 13     | 18     |
| 5 (W5)          | 10    | 13    | 12    | 19    | 15    | 11    | 12    | 14    | 12    | 15     | 17     |
| 6 (W6)          | 15    | 12    | 14    | 16    | 13    | 17    | 16    | 16    | 14    | 18     | 19     |
| 7 (W7)          | 14    | 13    | 15    | 17    | 12    | 13    | 14    | 15    | 12    | 16     | 14     |
| 8 (W8)          | 19    | 16    | 18    | 20    | 17    | 19    | 16    | 18    | 15    | 15     | 18     |
| 9 (W9)          | 17    | 18    | 12    | 14    | 16    | 15    | 14    | 17    | 21    | 15     | 18     |
| 10 (W10)        | 14    | 13    | 15    | 17    | 16    | 18    | 14    | 19    | 15    | 17     | 19     |
| 11 (W11)        | 15    | 13    | 16    | 17    | 11    | 13    | 14    | 15    | 19    | 21     | 13     |

##### Objective Function

\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For every store $j \in J$,
   \[
   \sum_{i=1}^{11} x_{ij} = d_j
   \]
2. **Warehouse capacity:**  
   For every warehouse $i \in I$,
   \[
   \sum_{j=1}^{11} x_{ij} \leq C_i y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i, j
   \]
   \[
   y_i \in \{0,1\} \quad \forall i
   \]

##### All parameters and data are as retrieved above, with indices matching the original CSV files.