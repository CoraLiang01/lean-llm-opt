##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$ (Suppliers/facilities)
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$ (Stores/customers)

- Demand for each store:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

  (Assume mapping: Customer_1 = CLARINDA, Customer_2 = FORT MADISON, Customer_3 = SIOUX CITY, Customer_4 = TOLEDO, Customer_5 = BANCROFT)

  So,
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Fixed costs for each supplier:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs $c_{ij}$ (per unit from supplier $i$ to store $j$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for each supplier's total shipment).

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
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameters (full data)

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $d_{\text{CLARINDA}} = 2397$, $d_{\text{FORT MADISON}} = 1889$, $d_{\text{SIOUX CITY}} = 2518$, $d_{\text{TOLEDO}} = 3218$, $d_{\text{BANCROFT}} = 1813$
- $f_{\text{MOUNT AYR}} = 96.58$, $f_{\text{WAUKEE}} = 94.06$, $f_{\text{WAVERLY}} = 94.37$, $f_{\text{PELLA}} = 82.88$, $f_{\text{DES MOINES}} = 94.96$
- $c_{ij}$ as in the table above
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