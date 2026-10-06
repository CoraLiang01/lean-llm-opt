##### Decision Variables

$x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (set of stores/customers)
- $f_i$: Fixed cost for opening supplier $i$ (see table below)
- $d_j$: Demand at store $j$ (see table below)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (see matrix below)
- $M = \sum_{j \in J} d_j$ (sufficiently large constant for linking constraints)

###### Fixed Costs ($f_i$)

| Supplier (Facility) | Fixed Cost |
|---------------------|------------|
| MOUNT AYR           | 96.58      |
| WAUKEE              | 94.06      |
| WAVERLY             | 94.37      |
| PELLA               | 82.88      |
| DES MOINES          | 94.96      |

###### Demands ($d_j$)

| Store (Customer) | Demand |
|------------------|--------|
| CLARINDA         | 2397   |
| FORT MADISON     | 1889   |
| SIOUX CITY       | 2518   |
| TOLEDO           | 3218   |
| BANCROFT         | 1813   |

Total demand $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

###### Transportation Costs ($c_{ij}$)

| Supplier (Facility) | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE              | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY             | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA               | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES          | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction:** Each store's demand must be met exactly.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier Activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (where $M = 11835$)

3. **Variable Domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where all parameters ($f_i$, $d_j$, $c_{ij}$) are as listed above, and all sets and indices are preserved from the original data.