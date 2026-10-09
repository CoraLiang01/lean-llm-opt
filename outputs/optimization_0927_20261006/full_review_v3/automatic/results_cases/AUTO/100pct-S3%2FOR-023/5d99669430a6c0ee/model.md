##### Sets

- Suppliers (Facilities): $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores (Customers): $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- Store locations for transportation costs: $S = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$

##### Parameters

- Demand at each store:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Fixed opening cost for each supplier:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation cost per unit from each supplier to each store location:

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to store $j$. (Assume mapping of stores to locations as needed; if each customer is at a unique location, use the above costs accordingly.)

- Total demand upper bound: $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (where $M = 11835$ is the sum of all demands)

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters (full vectors and matrices):

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- $d = [2397, 1889, 2518, 3218, 1813]$
- $f = [96.58, 94.06, 94.37, 82.88, 94.96]$
- $C =$
\[
\begin{array}{l|ccccc}
 & \text{CLARINDA} & \text{FORT MADISON} & \text{SIOUX CITY} & \text{TOLEDO} & \text{BANCROFT} \\
\hline
\text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
\text{WAUKEE} & 15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
\text{WAVERLY} & 2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
\text{PELLA} & 1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
\text{DES MOINES} & 1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{array}
\]

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where all parameters and sets are as listed above, and $c_{ij}$ is taken from the transportation cost matrix for each supplier-store pair.