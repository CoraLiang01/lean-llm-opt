##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier (facility) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$ (continuous).

$y_i \in \{0,1\}$: whether supplier (facility) $i$ is activated (open), for all $i \in I$ (binary).

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$ (set of stores/customers)
- $f_i$: fixed cost for opening supplier $i$ (from fixed_cost.csv)
- $d_j$: demand at store $j$ (from demand.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv)
- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking constraints)

###### Fixed Costs ($f_i$):

- $f_{\text{MOUNT AYR}} = 96.58$
- $f_{\text{WAUKEE}} = 94.06$
- $f_{\text{WAVERLY}} = 94.37$
- $f_{\text{PELLA}} = 82.88$
- $f_{\text{DES MOINES}} = 94.96$

###### Demands ($d_j$):

- $d_{\text{Customer\_1}} = 2397$
- $d_{\text{Customer\_2}} = 1889$
- $d_{\text{Customer\_3}} = 2518$
- $d_{\text{Customer\_4}} = 3218$
- $d_{\text{Customer\_5}} = 1813$

###### Transportation Costs ($c_{ij}$):

|                | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|----------------|------------|------------|------------|------------|------------|
| MOUNT AYR      | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE         | 15.13      | 1.5        | 1.43       | 27.88      | 90.69      |
| WAVERLY        | 2.34       | 349.34     | 246.6      | 41.3       | 78.73      |
| PELLA          | 1181.6     | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES     | 1030.8     | 43.48      | 932.43     | 55.39      | 103.84     |

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

2. **Supplier activation (linking):**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = 11835$.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model Data

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$
- $f_i$ as above
- $d_j$ as above
- $c_{ij}$ as above
- $M = 11835$

All parameters, vectors, and matrices are included as retrieved from the CSV files.