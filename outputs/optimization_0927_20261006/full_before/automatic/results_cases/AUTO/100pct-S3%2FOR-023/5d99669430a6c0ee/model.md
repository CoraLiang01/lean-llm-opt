##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- Suppliers (Facilities): $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores (Customers): $J = \{\text{SIOUX CITY}, \text{CLARINDA}, \text{FORT MADISON}, \text{TOLEDO}, \text{BANCROFT}\}$

- Fixed costs $f_i$ (Current Period):

  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs $c_{ij}$ (Current Period):

  | Supplier $\downarrow$ \ Store $\rightarrow$ | SIOUX CITY | CLARINDA | FORT MADISON | TOLEDO | BANCROFT |
  |---------------------------------------------|------------|----------|--------------|--------|----------|
  | MOUNT AYR                                   | 20.07      | 694.68   | 17.48        | 199.02 | 1685.53  |
  | WAUKEE                                      | 1.43       | 15.13    | 1.5          | 27.88  | 90.69    |
  | WAVERLY                                     | 246.6      | 2.34     | 349.34       | 41.3   | 78.73    |
  | PELLA                                       | 1646.36    | 1181.6   | 1458.53      | 1924.55| 38.93    |
  | DES MOINES                                  | 932.43     | 1030.8   | 43.48        | 55.39  | 103.84   |

- Demand $d_j$ (Current Period):

  - $d_{\text{SIOUX CITY}} = 2397$
  - $d_{\text{CLARINDA}} = 1889$
  - $d_{\text{FORT MADISON}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for total shipments from any supplier, since there are no explicit supplier capacity limits).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (no shipments from closed suppliers):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model (with all parameters)

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where:

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{SIOUX CITY}, \text{CLARINDA}, \text{FORT MADISON}, \text{TOLEDO}, \text{BANCROFT}\}$
- $f_i$ and $c_{ij}$ as specified above
- $d_j$ as specified above
- $M = 11835$

All data is preserved as in the original CSVs.