##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i$ to store (customer) $j$, for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise, for all $i \in I$.

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of suppliers/facilities)
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$ (set of stores/customers)
- Fixed costs $f_i$:

  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demands $d_j$:

  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Transportation costs $c_{ij}$ (per unit):

  | From \ To    | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
  |--------------|------------|------------|------------|------------|------------|
  | MOUNT AYR    | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
  | WAUKEE       | 15.13      | 1.5        | 1.43       | 27.88      | 90.69      |
  | WAVERLY      | 2.34       | 349.34     | 246.6      | 41.3       | 78.73      |
  | PELLA        | 1181.6     | 1458.53    | 1646.36    | 1924.55    | 38.93      |
  | DES MOINES   | 1030.8     | 43.48      | 932.43     | 55.39      | 103.84     |

- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
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
   (Inactive suppliers cannot ship any goods.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (explicit):

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$
- $f_{\text{MOUNT AYR}} = 96.58$, $f_{\text{WAUKEE}} = 94.06$, $f_{\text{WAVERLY}} = 94.37$, $f_{\text{PELLA}} = 82.88$, $f_{\text{DES MOINES}} = 94.96$
- $d_{\text{Customer\_1}} = 2397$, $d_{\text{Customer\_2}} = 1889$, $d_{\text{Customer\_3}} = 2518$, $d_{\text{Customer\_4}} = 3218$, $d_{\text{Customer\_5}} = 1813$
- $c_{ij}$ as in the table above
- $M = 11835$

##### Model Summary

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

All parameters, sets, and coefficients are as retrieved and listed above.