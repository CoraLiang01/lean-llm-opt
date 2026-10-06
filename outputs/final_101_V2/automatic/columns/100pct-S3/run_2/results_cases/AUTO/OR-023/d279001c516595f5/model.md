##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$ (set of stores/customers)
- $f_i$: Fixed cost for opening supplier $i$ (from fixed_cost.csv, current period)
- $d_j$: Demand at store $j$ (from demand.csv, current period)
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to store $j$ (from transportation_costs.csv, current period)
- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking constraints)

###### Fixed Costs (current period)
- $f_{\text{MOUNT AYR}} = 96.58$
- $f_{\text{WAUKEE}} = 94.06$
- $f_{\text{WAVERLY}} = 94.37$
- $f_{\text{PELLA}} = 82.88$
- $f_{\text{DES MOINES}} = 94.96$

###### Demands (current period)
- $d_{\text{Customer\_1}} = 2397$
- $d_{\text{Customer\_2}} = 1889$
- $d_{\text{Customer\_3}} = 2518$
- $d_{\text{Customer\_4}} = 3218$
- $d_{\text{Customer\_5}} = 1813$

###### Transportation Costs (current period, $c_{ij}$)

| Supplier $\downarrow$ \ Customer $\rightarrow$ | Customer_1 (SIOUX CITY) | Customer_2 (CLARINDA) | Customer_3 (FORT MADISON) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|-----------------------------------------------|-------------------------|-----------------------|---------------------------|---------------------|-----------------------|
| MOUNT AYR                                    | 20.07                   | 694.68                | 17.48                     | 199.02              | 1685.53               |
| WAUKEE                                       | 1.43                    | 15.13                 | 1.5                       | 27.88               | 90.69                 |
| WAVERLY                                      | 246.6                   | 2.34                  | 349.34                    | 41.3                | 78.73                 |
| PELLA                                        | 1646.36                 | 1181.6                | 1458.53                   | 1924.55             | 38.93                 |
| DES MOINES                                   | 932.43                  | 1030.8                | 43.48                     | 55.39               | 103.84                |

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

---

###### All parameters (current period, as retrieved):

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$
- $f_i$:
    - MOUNT AYR: 96.58
    - WAUKEE: 94.06
    - WAVERLY: 94.37
    - PELLA: 82.88
    - DES MOINES: 94.96
- $d_j$:
    - Customer_1: 2397
    - Customer_2: 1889
    - Customer_3: 2518
    - Customer_4: 3218
    - Customer_5: 1813
- $c_{ij}$:

|             | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|-------------|------------|------------|------------|------------|------------|
| MOUNT AYR   | 20.07      | 694.68     | 17.48      | 199.02     | 1685.53    |
| WAUKEE      | 1.43       | 15.13      | 1.5        | 27.88      | 90.69      |
| WAVERLY     | 246.6      | 2.34       | 349.34     | 41.3       | 78.73      |
| PELLA       | 1646.36    | 1181.6     | 1458.53    | 1924.55    | 38.93      |
| DES MOINES  | 932.43     | 1030.8     | 43.48      | 55.39      | 103.84     |

- $M = 11835$

---

This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost (fixed + transportation), using the exact data provided.