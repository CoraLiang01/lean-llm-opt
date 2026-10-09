##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (Suppliers)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (Stores)
- Store demands:
  - $d_{Customer_1} = 2397$
  - $d_{Customer_2} = 1889$
  - $d_{Customer_3} = 2518$
  - $d_{Customer_4} = 3218$
  - $d_{Customer_5} = 1813$
- Supplier fixed costs:
  - $f_{MOUNT AYR} = 96.58$
  - $f_{WAUKEE} = 94.06$
  - $f_{WAVERLY} = 94.37$
  - $f_{PELLA} = 82.88$
  - $f_{DES MOINES} = 94.96$
- Transportation costs per unit from supplier $i$ to store $j$ (matrix $c_{ij}$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

(Assuming: Customer_1 = CLARINDA, Customer_2 = FORT MADISON, Customer_3 = SIOUX CITY, Customer_4 = TOLEDO, Customer_5 = BANCROFT)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$
3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Parameter Listing

- Suppliers $I$: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Stores $J$: Customer_1 (CLARINDA), Customer_2 (FORT MADISON), Customer_3 (SIOUX CITY), Customer_4 (TOLEDO), Customer_5 (BANCROFT)
- Demands $d_j$:
  - $d_{Customer_1}$ = 2397
  - $d_{Customer_2}$ = 1889
  - $d_{Customer_3}$ = 2518
  - $d_{Customer_4}$ = 3218
  - $d_{Customer_5}$ = 1813
- Fixed costs $f_i$:
  - $f_{MOUNT AYR}$ = 96.58
  - $f_{WAUKEE}$ = 94.06
  - $f_{WAVERLY}$ = 94.37
  - $f_{PELLA}$ = 82.88
  - $f_{DES MOINES}$ = 94.96
- Transportation costs $c_{ij}$:

| $c_{ij}$         | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|------------------|----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR        | 694.68               | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE           | 15.13                | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY          | 2.34                 | 349.34                    | 246.6                   | 41.3                | 78.73                 |
| PELLA            | 1181.6               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES       | 1030.8               | 43.48                     | 932.43                  | 55.39               | 103.84                |

- $M = 11835$

##### Complete Mathematical Model

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