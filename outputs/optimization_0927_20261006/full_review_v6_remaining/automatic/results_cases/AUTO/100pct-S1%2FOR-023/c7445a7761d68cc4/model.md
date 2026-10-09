##### Sets

Let  
$F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$: set of suppliers (facility locations)  
$S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$: set of stores (customers)  
$C = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$: set of demand points (corresponding to stores, see mapping below)

##### Parameters

- Demands (from demand.csv):

| Customer      | Demand |
|---------------|--------|
| Customer_1    | 2397   |
| Customer_2    | 1889   |
| Customer_3    | 2518   |
| Customer_4    | 3218   |
| Customer_5    | 1813   |

Let $d_j$ be the demand for customer $j$.

- Fixed costs (from fixed_cost.csv):

| Supplier      | Fixed Cost |
|---------------|------------|
| MOUNT AYR     | 96.58      |
| WAUKEE        | 94.06      |
| WAVERLY       | 94.37      |
| PELLA         | 82.88      |
| DES MOINES    | 94.96      |

Let $f_i$ be the fixed cost for supplier $i$.

- Transportation costs (from transportation_costs.csv):

Let $c_{ij}$ be the per-unit transportation cost from supplier $i$ to store $j$:

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in F$ to store $j \in S$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise

##### Objective Function

\[
\min \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met.

   For each store $j \in S$ (corresponding to customer $j$):

   \[
   \sum_{i \in F} x_{ij} = d_j \qquad \forall j \in S
   \]

   (Assume mapping: CLARINDA $\to$ Customer_1, FORT MADISON $\to$ Customer_2, SIOUX CITY $\to$ Customer_3, TOLEDO $\to$ Customer_4, BANCROFT $\to$ Customer_5.)

2. **Supplier activation:** No shipments from inactive suppliers.

   For each supplier $i \in F$:

   \[
   \sum_{j \in S} x_{ij} \leq M \cdot y_i \qquad \forall i \in F
   \]

   where $M = \sum_{j \in S} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**

   \[
   x_{ij} \geq 0 \qquad \forall i \in F,\, j \in S
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in F
   \]

##### Parameters (explicit):

- $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
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
\min\quad & \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in F} x_{ij} = d_j \qquad \forall j \in S \\
& \sum_{j \in S} x_{ij} \leq 11835\, y_i \qquad \forall i \in F \\
& x_{ij} \geq 0 \qquad \forall i \in F,\, j \in S \\
& y_i \in \{0,1\} \qquad \forall i \in F
\end{align*}
\]

with all parameters and sets as specified above.