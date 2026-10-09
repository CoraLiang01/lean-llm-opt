##### Sets

Let $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$ be the set of suppliers.

Let $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$ be the set of stores.

Let $K = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$ be the set of demand points (for completeness, though store names and customer names may correspond).

##### Parameters

- Demand for each customer:
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

Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to store $j$ as given above.

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store must receive exactly its demand. (Assuming each store corresponds to a customer, and demand is mapped accordingly.)

Let us map:
- CLARINDA $\rightarrow$ Customer\_1 ($d_1 = 2397$)
- FORT MADISON $\rightarrow$ Customer\_2 ($d_2 = 1889$)
- SIOUX CITY $\rightarrow$ Customer\_3 ($d_3 = 2518$)
- TOLEDO $\rightarrow$ Customer\_4 ($d_4 = 3218$)
- BANCROFT $\rightarrow$ Customer\_5 ($d_5 = 1813$)

So, for each $j \in J$:
\[
\sum_{i \in I} x_{ij} = d_j
\]
where $d_j$ is the demand for store $j$ as mapped above.

2. **Supplier activation:** No shipments from inactive suppliers.
\[
\sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
\]
where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

3. **Variable domains:**
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

##### Full Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

##### Parameters (explicit):

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
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
- $c_{ij}$ as in the table above.
- $M = 11835$

This model determines which suppliers to activate and how much each should ship to each store to minimize total cost while meeting all store demands.