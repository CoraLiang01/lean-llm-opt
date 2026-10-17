##### Parameters

Let:
- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$: set of suppliers (indexed by $i$)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$: set of stores (indexed by $j$)
- $d_j$: demand at store $j$ (from demand.csv)
- $f_i$: fixed cost to activate supplier $i$ (from fixed_cost.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv)
- $M = \sum_{j \in J} d_j$ (a valid upper bound for total shipments from any supplier, since there are no supplier capacity limits)

From the data:

Demands (in order of stores as per transportation_costs.csv columns):
- $d_{CLARINDA} = 2397$
- $d_{FORT MADISON} = 1889$
- $d_{SIOUX CITY} = 2518$
- $d_{TOLEDO} = 3218$
- $d_{BANCROFT} = 1813$

Supplier fixed costs:
- $f_{MOUNT AYR} = 96.58$
- $f_{WAUKEE} = 94.06$
- $f_{WAVERLY} = 94.37$
- $f_{PELLA} = 82.88$
- $f_{DES MOINES} = 94.96$

Transportation costs $c_{ij}$ (rows: suppliers, columns: stores):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

$M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to store $j$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Parameter Listing

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$
- $d_{CLARINDA} = 2397$, $d_{FORT MADISON} = 1889$, $d_{SIOUX CITY} = 2518$, $d_{TOLEDO} = 3218$, $d_{BANCROFT} = 1813$
- $f_{MOUNT AYR} = 96.58$, $f_{WAUKEE} = 94.06$, $f_{WAVERLY} = 94.37$, $f_{PELLA} = 82.88$, $f_{DES MOINES} = 94.96$
- $c_{ij}$ as in the table above
- $M = 11835$

##### Model Summary

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

with all parameters as listed above.