##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from facility (supplier) $i$ to customer (store) $j$, for all $i \in I$, $j \in J$ (continuous).

$y_i \in \{0,1\}$: 1 if facility (supplier) $i$ is activated (open), 0 otherwise.

---

##### Parameters

- $I = \{$MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES$\}$ (set of facilities/suppliers)
- $J = \{$Customer\_1, Customer\_2, Customer\_3, Customer\_4, Customer\_5$\}$ (set of customers/stores)

- Demand $d_j$ for each customer $j$:
  - $d_{Customer\_1} = 2397$
  - $d_{Customer\_2} = 1889$
  - $d_{Customer\_3} = 2518$
  - $d_{Customer\_4} = 3218$
  - $d_{Customer\_5} = 1813$

- Fixed cost $f_i$ for each facility $i$:
  - $f_{MOUNT AYR} = 96.58$
  - $f_{WAUKEE} = 94.06$
  - $f_{WAVERLY} = 94.37$
  - $f_{PELLA} = 82.88$
  - $f_{DES MOINES} = 94.96$

- Transportation cost $c_{ij}$ for each facility $i$ and customer $j$:

| $c_{ij}$           | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|--------------------|------------|------------|------------|------------|------------|
| MOUNT AYR          | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE             | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY            | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA              | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES         | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for total shipments from any facility, since there are no explicit capacity limits).

---

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

---

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   That is,
   - $\sum_{i \in I} x_{i,Customer\_1} = 2397$
   - $\sum_{i \in I} x_{i,Customer\_2} = 1889$
   - $\sum_{i \in I} x_{i,Customer\_3} = 2518$
   - $\sum_{i \in I} x_{i,Customer\_4} = 3218$
   - $\sum_{i \in I} x_{i,Customer\_5} = 1813$

2. **Facility activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   That is, if $y_i = 0$, then $x_{ij} = 0$ for all $j$.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
                  & \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
                  & x_{ij} \geq 0 \quad \forall i \in I, j \in J \\
                  & y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

Where all parameters ($c_{ij}$, $f_i$, $d_j$, $M$) and sets ($I$, $J$) are as listed above, with all identifiers and coefficients preserved from the original data.