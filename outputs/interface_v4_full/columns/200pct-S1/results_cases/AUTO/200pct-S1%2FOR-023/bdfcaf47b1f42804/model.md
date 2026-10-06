##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$ (set of suppliers)
- $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$ (set of stores)
- $f_i$: Fixed cost for opening supplier $i$ (see table below)
- $d_j$: Demand at store $j$ (see table below)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (see matrix below)
- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking constraints)

###### Fixed Costs

| Supplier      | $f_i$   |
|---------------|---------|
| MOUNT AYR     | 96.58   |
| WAUKEE        | 94.06   |
| WAVERLY       | 94.37   |
| PELLA         | 82.88   |
| DES MOINES    | 94.96   |

###### Store Demands

| Store         | $d_j$   |
|---------------|---------|
| CLARINDA      | 2397    |
| FORT MADISON  | 1889    |
| SIOUX CITY    | 2518    |
| TOLEDO        | 3218    |
| BANCROFT      | 1813    |

###### Transportation Cost Matrix $c_{ij}$

| Supplier   | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|------------|----------|--------------|------------|--------|----------|
| MOUNT AYR  | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE     | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY    | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA      | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

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

2. **Supplier Activation (Linking):**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = 11835$.

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I
\end{align*}
\]

where all parameters ($f_i$, $d_j$, $c_{ij}$, $M$) and sets ($I$, $J$) are as specified above, with all identifiers and coefficients preserved from the original data.