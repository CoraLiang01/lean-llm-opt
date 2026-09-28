Let:
- $F$ = set of suppliers = {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES}
- $S$ = set of stores = {CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT}
- $C$ = set of customers = {Customer_1, Customer_2, Customer_3, Customer_4, Customer_5}
- $f_i$ = fixed cost of opening supplier $i \in F$
- $t_{ij}$ = transportation cost per unit from supplier $i \in F$ to store $j \in S$
- $d_k$ = demand of customer $k \in C$
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to store $j$

##### Data

- Fixed costs ($f_i$):

| Supplier      | Fixed Cost |
|---------------|-----------|
| MOUNT AYR     | 96.58     |
| WAUKEE        | 94.06     |
| WAVERLY       | 94.37     |
| PELLA         | 82.88     |
| DES MOINES    | 94.96     |

- Transportation costs ($t_{ij}$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Customer demands ($d_k$):

| Customer     | Demand |
|--------------|--------|
| Customer_1   | 2397   |
| Customer_2   | 1889   |
| Customer_3   | 2518   |
| Customer_4   | 3218   |
| Customer_5   | 1813   |

##### Mathematical Model

Minimize total cost:
$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} t_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction: (Assume each customer corresponds to a store in the same order as the files; i.e., Customer_1 = CLARINDA, ..., Customer_5 = BANCROFT)
$$
\sum_{i \in F} x_{ij} \geq d_k \qquad \forall j \in S, \forall k \in C \text{ with } j \text{ corresponding to } k
$$
Explicitly:
\begin{align*}
\sum_{i \in F} x_{i,\text{CLARINDA}} &\geq 2397 \\
\sum_{i \in F} x_{i,\text{FORT MADISON}} &\geq 1889 \\
\sum_{i \in F} x_{i,\text{SIOUX CITY}} &\geq 2518 \\
\sum_{i \in F} x_{i,\text{TOLEDO}} &\geq 3218 \\
\sum_{i \in F} x_{i,\text{BANCROFT}} &\geq 1813 \\
\end{align*}

2. Supplier activation:
$$
\sum_{j \in S} x_{ij} \leq M y_i \qquad \forall i \in F
$$
where $M$ is a sufficiently large constant (e.g., $M = \sum_k d_k$).

3. Variable domains:
$$
x_{ij} \geq 0 \quad \text{and integer} \qquad \forall i \in F, \forall j \in S \\
y_i \in \{0,1\} \qquad \forall i \in F
$$

##### Complete Numerical Formulation

Minimize
$$
96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}} \\
+ \sum_{i \in F} \sum_{j \in S} t_{ij} x_{ij}
$$
where $t_{ij}$ are as in the table above.

Subject to:
\begin{align*}
x_{\text{MOUNT AYR},\text{CLARINDA}} + x_{\text{WAUKEE},\text{CLARINDA}} + x_{\text{WAVERLY},\text{CLARINDA}} + x_{\text{PELLA},\text{CLARINDA}} + x_{\text{DES MOINES},\text{CLARINDA}} &\geq 2397 \\
x_{\text{MOUNT AYR},\text{FORT MADISON}} + x_{\text{WAUKEE},\text{FORT MADISON}} + x_{\text{WAVERLY},\text{FORT MADISON}} + x_{\text{PELLA},\text{FORT MADISON}} + x_{\text{DES MOINES},\text{FORT MADISON}} &\geq 1889 \\
x_{\text{MOUNT AYR},\text{SIOUX CITY}} + x_{\text{WAUKEE},\text{SIOUX CITY}} + x_{\text{WAVERLY},\text{SIOUX CITY}} + x_{\text{PELLA},\text{SIOUX CITY}} + x_{\text{DES MOINES},\text{SIOUX CITY}} &\geq 2518 \\
x_{\text{MOUNT AYR},\text{TOLEDO}} + x_{\text{WAUKEE},\text{TOLEDO}} + x_{\text{WAVERLY},\text{TOLEDO}} + x_{\text{PELLA},\text{TOLEDO}} + x_{\text{DES MOINES},\text{TOLEDO}} &\geq 3218 \\
x_{\text{MOUNT AYR},\text{BANCROFT}} + x_{\text{WAUKEE},\text{BANCROFT}} + x_{\text{WAVERLY},\text{BANCROFT}} + x_{\text{PELLA},\text{BANCROFT}} + x_{\text{DES MOINES},\text{BANCROFT}} &\geq 1813 \\
\end{align*}

For each supplier $i$:
$$
x_{i,\text{CLARINDA}} + x_{i,\text{FORT MADISON}} + x_{i,\text{SIOUX CITY}} + x_{i,\text{TOLEDO}} + x_{i,\text{BANCROFT}} \leq M y_i
$$

Variable domains:
$$
x_{ij} \geq 0 \text{ and integer} \qquad \forall i, j \\
y_i \in \{0,1\} \qquad \forall i
$$

All coefficients and identifiers are as retrieved and shown above.