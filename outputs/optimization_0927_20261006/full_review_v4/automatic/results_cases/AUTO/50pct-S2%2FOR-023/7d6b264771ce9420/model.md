##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (operational), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (Suppliers)
- $J = \{$CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT$\}$ (Stores)
- $K = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (Demand entries, mapped to stores as below)
- $d_j$: Demand at store $j$ (see below)
- $f_i$: Fixed cost for activating supplier $i$ (see below)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (see below)
- $M = \sum_{j \in J} d_j$ (A sufficiently large constant for linking $x_{ij}$ and $y_i$)

###### Demand Data

| Store         | Demand |
|---------------|--------|
| CLARINDA      | 2397   |
| FORT MADISON  | 1889   |
| SIOUX CITY    | 2518   |
| TOLEDO        | 3218   |
| BANCROFT      | 1813   |

###### Fixed Cost Data

| Supplier      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

###### Transportation Cost Matrix ($c_{ij}$)

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

$M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier Activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (as retrieved)

- Suppliers $I$: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Stores $J$: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- Demands $d_j$:
  - CLARINDA: 2397
  - FORT MADISON: 1889
  - SIOUX CITY: 2518
  - TOLEDO: 3218
  - BANCROFT: 1813
- Fixed costs $f_i$:
  - MOUNT AYR: 96.58
  - WAUKEE: 94.06
  - WAVERLY: 94.37
  - PELLA: 82.88
  - DES MOINES: 94.96
- Transportation costs $c_{ij}$:
  - MOUNT AYR: [694.68, 17.48, 20.07, 199.02, 1685.53]
  - WAUKEE: [15.13, 1.5, 1.43, 27.88, 90.69]
  - WAVERLY: [2.34, 349.34, 246.6, 41.3, 78.73]
  - PELLA: [1181.6, 1458.53, 1646.36, 1924.55, 38.93]
  - DES MOINES: [1030.8, 43.48, 932.43, 55.39, 103.84]
- $M = 11835$

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

with all parameters as listed above.