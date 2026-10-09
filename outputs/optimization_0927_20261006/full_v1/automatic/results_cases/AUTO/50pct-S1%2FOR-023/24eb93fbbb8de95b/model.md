##### Parameters

- Suppliers $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
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
- Transportation costs $c_{ij}$ (supplier $i$ to store $j$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to store $j$ (continuous)
- $y_i \in \{0,1\}$: $1$ if supplier $i$ is activated, $0$ otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where $c_{ij}$ and $f_i$ are as above.

##### Constraints

1. **Demand satisfaction:** For each store $j$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   where $d_j$ is the demand for store $j$ (see mapping below).

2. **Supplier activation:** For each supplier $i$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Mapping of Customers to Stores

Assume the following mapping for demand (since store names in transportation_costs.csv correspond to $J$):

- $\text{Customer}_1$ demand $2397$ → $\text{CLARINDA}$
- $\text{Customer}_2$ demand $1889$ → $\text{FORT MADISON}$
- $\text{Customer}_3$ demand $2518$ → $\text{SIOUX CITY}$
- $\text{Customer}_4$ demand $3218$ → $\text{TOLEDO}$
- $\text{Customer}_5$ demand $1813$ → $\text{BANCROFT}$

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{i,\text{CLARINDA}} = 2397 \\
& \sum_{i \in I} x_{i,\text{FORT MADISON}} = 1889 \\
& \sum_{i \in I} x_{i,\text{SIOUX CITY}} = 2518 \\
& \sum_{i \in I} x_{i,\text{TOLEDO}} = 3218 \\
& \sum_{i \in I} x_{i,\text{BANCROFT}} = 1813 \\
& \sum_{j \in J} x_{\text{MOUNT AYR},j} \leq 11835\, y_{\text{MOUNT AYR}} \\
& \sum_{j \in J} x_{\text{WAUKEE},j} \leq 11835\, y_{\text{WAUKEE}} \\
& \sum_{j \in J} x_{\text{WAVERLY},j} \leq 11835\, y_{\text{WAVERLY}} \\
& \sum_{j \in J} x_{\text{PELLA},j} \leq 11835\, y_{\text{PELLA}} \\
& \sum_{j \in J} x_{\text{DES MOINES},j} \leq 11835\, y_{\text{DES MOINES}} \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

##### Parameters (full vectors/matrices):

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- Demands: $d = [2397, 1889, 2518, 3218, 1813]$
- Fixed costs: $f = [96.58, 94.06, 94.37, 82.88, 94.96]$
- Transportation cost matrix $C =$
\[
\begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]
with rows in supplier order and columns in store order as above.

$M = 11835$.

This model determines which suppliers to activate and how much each should ship to each store to minimize total cost while meeting all store demands.