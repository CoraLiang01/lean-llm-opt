##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from facility (supplier) $i \in I$ to customer (branch) $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if facility $i$ is opened (operational), 0 otherwise.

##### Parameters

- Facilities (Suppliers): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Customers (Branches): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

- Demand for each customer:
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$

- Fixed opening cost for each facility:
  - $f_{\text{S1}} = 97.65$
  - $f_{\text{S2}} = 99.76$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$

- Transportation cost per unit from each facility to each customer:
  - $c_{\text{S1},\text{C1}} = 150.74$, $c_{\text{S1},\text{C2}} = 0.02$, $c_{\text{S1},\text{C3}} = 49.13$, $c_{\text{S1},\text{C4}} = 2080.15$, $c_{\text{S1},\text{C5}} = 426.4$
  - $c_{\text{S2},\text{C1}} = 233.05$, $c_{\text{S2},\text{C2}} = 97.73$, $c_{\text{S2},\text{C3}} = 49.84$, $c_{\text{S2},\text{C4}} = 1982.39$, $c_{\text{S2},\text{C5}} = 23.96$
  - $c_{\text{S3},\text{C1}} = 55.68$, $c_{\text{S3},\text{C2}} = 935.61$, $c_{\text{S3},\text{C3}} = 4.03$, $c_{\text{S3},\text{C4}} = 73.09$, $c_{\text{S3},\text{C5}} = 525.32$
  - $c_{\text{S4},\text{C1}} = 1483.82$, $c_{\text{S4},\text{C2}} = 1801.08$, $c_{\text{S4},\text{C3}} = 112.16$, $c_{\text{S4},\text{C4}} = 816.05$, $c_{\text{S4},\text{C5}} = 107.01$
  - $c_{\text{S5},\text{C1}} = 1119.47$, $c_{\text{S5},\text{C2}} = 884.31$, $c_{\text{S5},\text{C3}} = 0.08$, $c_{\text{S5},\text{C4}} = 1544.95$, $c_{\text{S5},\text{C5}} = 543.67$

- Let $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (a valid upper bound for total shipments from any facility, since there are no explicit capacity limits).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility activation:**  
   For each facility $i \in I$,
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

##### Full Data Used

- Facilities: S1, S2, S3, S4, S5
- Customers: C1, C2, C3, C4, C5
- Demand: C1=143, C2=6, C3=10, C4=25, C5=3
- Fixed costs: S1=97.65, S2=99.76, S3=100.76, S4=105.32, S5=98.88
- Transportation cost matrix (rows: facilities, columns: customers):

|        | C1      | C2      | C3     | C4      | C5     |
|--------|---------|---------|--------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67 |

- $M = 187$

##### Model Summary

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

where all parameters and data are as listed above.