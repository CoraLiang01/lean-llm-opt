##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- Store demands:
  - $\text{Customer}_1$: $2397$
  - $\text{Customer}_2$: $1889$
  - $\text{Customer}_3$: $2518$
  - $\text{Customer}_4$: $3218$
  - $\text{Customer}_5$: $1813$
- Supplier fixed costs:
  - $\text{MOUNT AYR}$: $96.58$
  - $\text{WAUKEE}$: $94.06$
  - $\text{WAVERLY}$: $94.37$
  - $\text{PELLA}$: $82.88$
  - $\text{DES MOINES}$: $94.96$
- Transportation costs $c_{ij}$ (supplier $i$, store $j$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Let $d_j$ be the demand for store $j$ (mapping of Customer_k to store name must be clarified; for this model, assume $J$ and demand indices correspond in order).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where $c_{ij}$ is the transportation cost from supplier $i$ to store $j$, and $f_i$ is the fixed cost for supplier $i$.

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from inactive suppliers:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Parameter Listing

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $d = [2397, 1889, 2518, 3218, 1813]$
- $f = [96.58, 94.06, 94.37, 82.88, 94.96]$
- $C =$
  \[
  \begin{array}{l|ccccc}
  & \text{CLARINDA} & \text{FORT MADISON} & \text{SIOUX CITY} & \text{TOLEDO} & \text{BANCROFT} \\
  \hline
  \text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
  \text{WAUKEE} & 15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
  \text{WAVERLY} & 2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
  \text{PELLA} & 1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
  \text{DES MOINES} & 1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
  \end{array}
  \]
- $M = 11835$

##### Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where all parameters are as listed above.