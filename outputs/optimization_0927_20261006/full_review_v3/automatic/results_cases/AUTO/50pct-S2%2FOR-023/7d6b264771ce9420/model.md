##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (Suppliers/Facilities)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (Stores/Customers)
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
- Transportation costs per unit from supplier $i$ to store $j$ (as per mapping below):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

Assume the stores correspond to: 
- Customer_1 = CLARINDA
- Customer_2 = FORT MADISON
- Customer_3 = SIOUX CITY
- Customer_4 = TOLEDO
- Customer_5 = BANCROFT

So, $c_{ij}$ is as follows:

| $c_{ij}$         | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|------------------|----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR        | 694.68               | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE           | 15.13                | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY          | 2.34                 | 349.34                    | 246.6                   | 41.3                | 78.73                 |
| PELLA            | 1181.6               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES       | 1030.8               | 43.48                     | 932.43                  | 55.39               | 103.84                |

Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### All Parameters (as retrieved)

- Suppliers $I$: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Stores $J$: Customer_1 (CLARINDA), Customer_2 (FORT MADISON), Customer_3 (SIOUX CITY), Customer_4 (TOLEDO), Customer_5 (BANCROFT)
- Demands:
  - Customer_1: 2397
  - Customer_2: 1889
  - Customer_3: 2518
  - Customer_4: 3218
  - Customer_5: 1813
- Fixed costs:
  - MOUNT AYR: 96.58
  - WAUKEE: 94.06
  - WAVERLY: 94.37
  - PELLA: 82.88
  - DES MOINES: 94.96
- Transportation costs $c_{ij}$ (see table above)
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