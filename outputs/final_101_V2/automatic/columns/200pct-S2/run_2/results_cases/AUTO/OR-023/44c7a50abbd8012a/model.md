##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from facility (supplier) $i \in I$ to customer (store) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if facility (supplier) $i$ is operational (open), 0 otherwise.

##### Parameters

- $I = \{\text{F1 (MOUNT AYR)},\ \text{F2 (WAUKEE)},\ \text{F3 (WAVERLY)},\ \text{F4 (PELLA)},\ \text{F5 (DES MOINES)}\}$
- $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$

- Demands $d_j$ for each customer $j$:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Fixed costs $f_i$ for each facility $i$:
  - $f_{\text{F1}} = 96.58$
  - $f_{\text{F2}} = 94.06$
  - $f_{\text{F3}} = 94.37$
  - $f_{\text{F4}} = 82.88$
  - $f_{\text{F5}} = 94.96$

- Transportation costs $c_{ij}$ (per unit from facility $i$ to customer $j$):

| Facility $\backslash$ Customer | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|-------------------------------|------------|------------|------------|------------|------------|
| F1 (MOUNT AYR)                | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| F2 (WAUKEE)                   | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| F3 (WAVERLY)                  | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| F4 (PELLA)                    | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| F5 (DES MOINES)               | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for total shipments from any facility, since there are no explicit capacity limits).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model Data

- Facilities (Suppliers):  
  - F1: MOUNT AYR, $f_{F1} = 96.58$
  - F2: WAUKEE, $f_{F2} = 94.06$
  - F3: WAVERLY, $f_{F3} = 94.37$
  - F4: PELLA, $f_{F4} = 82.88$
  - F5: DES MOINES, $f_{F5} = 94.96$

- Customers (Stores) and Demands:
  - Customer_1: $d_1 = 2397$
  - Customer_2: $d_2 = 1889$
  - Customer_3: $d_3 = 2518$
  - Customer_4: $d_4 = 3218$
  - Customer_5: $d_5 = 1813$

- Transportation cost matrix $c_{ij}$:

|         | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|---------|------------|------------|------------|------------|------------|
| F1      | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| F2      | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| F3      | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| F4      | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| F5      | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- $M = 11835$

---

This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost (fixed + transportation), using the exact data provided.