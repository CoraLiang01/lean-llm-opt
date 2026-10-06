##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- Facilities (Suppliers): $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Customers (Stores): $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

- Fixed costs for each facility:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demand for each customer:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Transportation costs $c_{ij}$ (per unit) from facility $i$ to customer $j$:

| Facility \ Customer | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR           | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE              | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY             | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA               | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES          | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Mathematical Model

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

**Subject to:**

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (Vectors and Matrices)

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- $f = [96.58,\ 94.06,\ 94.37,\ 82.88,\ 94.96]$
- $d = [2397,\ 1889,\ 2518,\ 3218,\ 1813]$
- $C =$
\[
\begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]
- $M = 11835$

##### Summary

This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost (fixed + transportation), using the exact data as provided.