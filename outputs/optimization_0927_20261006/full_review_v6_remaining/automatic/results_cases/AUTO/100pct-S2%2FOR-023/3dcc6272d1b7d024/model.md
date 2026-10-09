##### Sets

Let  
$F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$: set of suppliers (facilities)  
$S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$: set of stores (customers)  
Let $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$: set of demand points (corresponding to stores above; mapping assumed 1-to-1 in order)

##### Parameters

- Demand for each store:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

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

Let $c_{is}$ denote the transportation cost per unit from supplier $i$ to store $s$ as above.

Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for each supplier's total shipments).

##### Decision Variables

- $x_{is} \geq 0$: quantity shipped from supplier $i \in F$ to store $s \in S$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise

##### Objective Function

\[
\min \sum_{i \in F} \sum_{s \in S} c_{is} x_{is} + \sum_{i \in F} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $s_j \in S$ (corresponding to $j \in J$),  
   \[
   \sum_{i \in F} x_{is_j} = d_j \qquad \forall j \in J
   \]
   (Each store's demand must be fully met.)

2. **Supplier activation:**  
   For each supplier $i \in F$,  
   \[
   \sum_{s \in S} x_{is} \leq M y_i
   \]
   (No shipments from inactive suppliers.)

3. **Variable domains:**  
   \[
   x_{is} \geq 0 \quad \forall i \in F,\, s \in S
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]

##### Parameters (full data)

- $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$
- Demands: $d_{\text{Customer\_1}} = 2397$, $d_{\text{Customer\_2}} = 1889$, $d_{\text{Customer\_3}} = 2518$, $d_{\text{Customer\_4}} = 3218$, $d_{\text{Customer\_5}} = 1813$
- Fixed costs: $f_{\text{MOUNT AYR}} = 96.58$, $f_{\text{WAUKEE}} = 94.06$, $f_{\text{WAVERLY}} = 94.37$, $f_{\text{PELLA}} = 82.88$, $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{is}$ as in the table above
- $M = 11835$

##### Model summary

\[
\begin{align*}
\min\ & \sum_{i \in F} \sum_{s \in S} c_{is} x_{is} + \sum_{i \in F} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in F} x_{is_j} = d_j \qquad \forall j \in J \\
& \sum_{s \in S} x_{is} \leq M y_i \qquad \forall i \in F \\
& x_{is} \geq 0 \qquad \forall i \in F,\, s \in S \\
& y_i \in \{0,1\} \qquad \forall i \in F
\end{align*}
\]