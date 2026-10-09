##### Decision Variables

$x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- Facilities (Suppliers):  
  $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$

- Stores (Customers):  
  $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$

- Fixed opening costs $f_i$ (from fixed_cost.csv):  
  \[
  \begin{align*}
  f_{\text{MOUNT AYR}} &= 96.58 \\
  f_{\text{WAUKEE}} &= 94.06 \\
  f_{\text{WAVERLY}} &= 94.37 \\
  f_{\text{PELLA}} &= 82.88 \\
  f_{\text{DES MOINES}} &= 94.96 \\
  \end{align*}
  \]

- Demand $d_j$ (from demand.csv):  
  \[
  \begin{align*}
  d_{\text{Customer\_1}} &= 2397 \\
  d_{\text{Customer\_2}} &= 1889 \\
  d_{\text{Customer\_3}} &= 2518 \\
  d_{\text{Customer\_4}} &= 3218 \\
  d_{\text{Customer\_5}} &= 1813 \\
  \end{align*}
  \]

- Transportation costs $c_{ij}$ (from transportation_costs.csv):  
  (Rows: facilities, Columns: stores. The mapping from store names to cost columns is as follows:  
  Customer_1 = CLARINDA, Customer_2 = FORT MADISON, Customer_3 = SIOUX CITY, Customer_4 = TOLEDO, Customer_5 = BANCROFT)

  \[
  \begin{array}{l|ccccc}
  & \text{Customer\_1} & \text{Customer\_2} & \text{Customer\_3} & \text{Customer\_4} & \text{Customer\_5} \\
  \hline
  \text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
  \text{WAUKEE} & 15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
  \text{WAVERLY} & 2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
  \text{PELLA} & 1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
  \text{DES MOINES} & 1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
  \end{array}
  \]

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking constraints).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation (linking):**  
   For each supplier $i \in I$,
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

##### Complete Model (with all parameters)

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
\end{align*}
\]

Where:

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$
- $f_i$ and $c_{ij}$ as given above
- $d_j$ as given above
- $M = 11835$

All data and identifiers are preserved as in the original files.