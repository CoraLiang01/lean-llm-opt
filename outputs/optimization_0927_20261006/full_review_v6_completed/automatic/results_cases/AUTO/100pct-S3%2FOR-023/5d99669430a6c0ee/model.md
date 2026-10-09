##### Sets and Indices

- Let $F = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$ be the set of suppliers, indexed by $i$.
- Let $S = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$ be the set of stores, indexed by $j$.

##### Parameters

- Store demands:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Supplier fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs $c_{ij}$ (supplier $i$ to store $j$):

| Supplier $\downarrow$ \ Store $\rightarrow$ | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|:-------------------------------------------:|:---------------------:|:-------------------------:|:----------------------:|:-------------------:|:---------------------:|
| MOUNT AYR                                  | 694.68                | 17.48                     | 20.07                  | 199.02              | 1685.53               |
| WAUKEE                                     | 15.13                 | 1.5                       | 1.43                   | 27.88               | 90.69                 |
| WAVERLY                                    | 2.34                  | 349.34                    | 246.6                  | 41.3                | 78.73                 |
| PELLA                                      | 1181.6                | 1458.53                   | 1646.36                | 1924.55             | 38.93                 |
| DES MOINES                                 | 1030.8                | 43.48                     | 932.43                 | 55.39               | 103.84                |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in F$ to store $j \in S$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met exactly.
   \[
   \sum_{i \in F} x_{ij} = d_j, \quad \forall j \in S
   \]

2. **Supplier activation:** No shipments from inactive suppliers.
   \[
   \sum_{j \in S} x_{ij} \leq M y_i, \quad \forall i \in F
   \]
   where $M = \sum_{j \in S} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters (full data)

- $F = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $S = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- Demands:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$
- Fixed costs:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{ij}$:

| $c_{ij}$           | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|--------------------|------------|------------|------------|------------|------------|
| MOUNT AYR          | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE             | 15.13      | 1.5        | 1.43       | 27.88      | 90.69      |
| WAVERLY            | 2.34       | 349.34     | 246.6      | 41.3       | 78.73      |
| PELLA              | 1181.6     | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES         | 1030.8     | 43.48      | 932.43     | 55.39      | 103.84     |

- $M = 11835$

##### Complete Model

\[
\begin{align*}
\min \quad & \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in F} x_{ij} = d_j, \quad \forall j \in S \\
& \sum_{j \in S} x_{ij} \leq M y_i, \quad \forall i \in F \\
& x_{ij} \geq 0, \quad \forall i \in F, j \in S \\
& y_i \in \{0,1\}, \quad \forall i \in F
\end{align*}
\]