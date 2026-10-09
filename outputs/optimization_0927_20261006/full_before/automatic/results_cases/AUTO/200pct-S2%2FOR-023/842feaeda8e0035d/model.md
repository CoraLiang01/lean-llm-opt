##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from facility (supplier) $i$ to customer (store) $j$, for all $i \in I$, $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if facility (supplier) $i$ is operational (open), 0 otherwise.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of facilities/suppliers)
- $J = \{$Customer_1, Customer_2, Customer_3, Customer_4, Customer_5$\}$ (set of customers/stores)

- Demands $d_j$ for each customer $j$:
  - $d_{Customer_1} = 2397$
  - $d_{Customer_2} = 1889$
  - $d_{Customer_3} = 2518$
  - $d_{Customer_4} = 3218$
  - $d_{Customer_5} = 1813$

- Fixed costs $f_i$ for each facility $i$:
  - $f_{MOUNT AYR} = 96.58$
  - $f_{WAUKEE} = 94.06$
  - $f_{WAVERLY} = 94.37$
  - $f_{PELLA} = 82.88$
  - $f_{DES MOINES} = 94.96$

- Transportation costs $c_{ij}$ (per unit from facility $i$ to customer $j$):

| $c_{ij}$         | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|------------------|------------|------------|------------|------------|------------|
| MOUNT AYR        | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE           | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY          | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA            | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES       | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for total shipments from any facility, since there are no explicit capacity limits).

---

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer (store) $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility activation:**  
   For each facility (supplier) $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (Inactive facilities cannot ship any goods.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

##### All Parameters (retrieved from CSVs)

- Facilities (Suppliers): MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Customers (Stores): Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- Demands: $d_{Customer_1} = 2397$, $d_{Customer_2} = 1889$, $d_{Customer_3} = 2518$, $d_{Customer_4} = 3218$, $d_{Customer_5} = 1813$
- Fixed Costs: $f_{MOUNT AYR} = 96.58$, $f_{WAUKEE} = 94.06$, $f_{WAVERLY} = 94.37$, $f_{PELLA} = 82.88$, $f_{DES MOINES} = 94.96$
- Transportation Costs: as in the table above
- $M = 11835$

---

This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost, using the exact data provided.