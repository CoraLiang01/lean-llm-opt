##### Sets

Let  
$F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$: set of suppliers (facilities)  
$S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$: set of stores (customers)  
$C = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$: set of demand points (corresponding to stores)

##### Parameters

- Demands (from demand.csv):  
  $d_{\text{Customer\_1}} = 2397$  
  $d_{\text{Customer\_2}} = 1889$  
  $d_{\text{Customer\_3}} = 2518$  
  $d_{\text{Customer\_4}} = 3218$  
  $d_{\text{Customer\_5}} = 1813$  

- Fixed costs (from fixed_cost.csv):  
  $f_{\text{MOUNT AYR}} = 96.58$  
  $f_{\text{WAUKEE}} = 94.06$  
  $f_{\text{WAVERLY}} = 94.37$  
  $f_{\text{PELLA}} = 82.88$  
  $f_{\text{DES MOINES}} = 94.96$  

- Transportation costs (from transportation_costs.csv):  
  (Rows: suppliers, Columns: stores)

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

where $c_{ij}$ is the transportation cost per unit from supplier $i$ to store $j$ (see table above), and $f_i$ is the fixed cost for supplier $i$.

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in S$, the total quantity received from all suppliers must meet its demand:
   \[
   \sum_{i \in F} x_{ij} = d_j, \quad \forall j \in S
   \]
   (where $d_j$ is the demand for store $j$; mapping from Customer_k to store $j$ as per data.)

2. **Supplier activation:**  
   A supplier can only ship if it is activated:
   \[
   \sum_{j \in S} x_{ij} \leq M y_i, \quad \forall i \in F
   \]
   where $M = \sum_{j \in S} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (the total demand across all stores).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Model

\[
\begin{align*}
\min\ & \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in F} x_{ij} = d_j, \quad \forall j \in S \\
& \sum_{j \in S} x_{ij} \leq 11835\, y_i, \quad \forall i \in F \\
& x_{ij} \geq 0,\quad \forall i \in F,\, j \in S \\
& y_i \in \{0,1\},\quad \forall i \in F
\end{align*}
\]

##### Parameter Summary

- $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- Demands: $d_{\text{CLARINDA}} = 2397$, $d_{\text{FORT MADISON}} = 1889$, $d_{\text{SIOUX CITY}} = 2518$, $d_{\text{TOLEDO}} = 3218$, $d_{\text{BANCROFT}} = 1813$
- Fixed costs: as above
- Transportation costs: as above
- $M = 11835$

All parameters are as retrieved from the CSV files, with all identifiers and values preserved.