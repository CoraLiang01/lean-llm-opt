##### Sets

- Suppliers (facilities): $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Stores (customers): $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

##### Parameters

- Demand for each store:
  - $d_{\text{Customer}_1} = 2397$
  - $d_{\text{Customer}_2} = 1889$
  - $d_{\text{Customer}_3} = 2518$
  - $d_{\text{Customer}_4} = 3218$
  - $d_{\text{Customer}_5} = 1813$

  (Assume mapping: $\text{Customer}_1 \rightarrow \text{CLARINDA}$, $\text{Customer}_2 \rightarrow \text{FORT MADISON}$, $\text{Customer}_3 \rightarrow \text{SIOUX CITY}$, $\text{Customer}_4 \rightarrow \text{TOLEDO}$, $\text{Customer}_5 \rightarrow \text{BANCROFT}$)

- Fixed cost for each supplier:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation cost per unit from each supplier to each store:

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where $c_{ij}$ is the transportation cost per unit from supplier $i$ to store $j$, and $f_i$ is the fixed cost for supplier $i$.

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met exactly.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters (full data)

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- $d_{\text{CLARINDA}} = 2397$
- $d_{\text{FORT MADISON}} = 1889$
- $d_{\text{SIOUX CITY}} = 2518$
- $d_{\text{TOLEDO}} = 3218$
- $d_{\text{BANCROFT}} = 1813$
- $f_{\text{MOUNT AYR}} = 96.58$
- $f_{\text{WAUKEE}} = 94.06$
- $f_{\text{WAVERLY}} = 94.37$
- $f_{\text{PELLA}} = 82.88$
- $f_{\text{DES MOINES}} = 94.96$
- $c_{ij}$ as in the table above
- $M = 11835$

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

with all parameters and sets as specified above.