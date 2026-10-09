##### Parameters

- Suppliers (indexed by $i$):  
  $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$

- Stores/Customers (indexed by $j$):  
  $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$

- Store locations (for transportation cost matrix columns):  
  $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

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

- Transportation cost per unit from supplier $i$ to store $s$ ($c_{is}$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Decision Variables

- $x_{is} \geq 0$: quantity shipped from supplier $i \in I$ to store $s \in S$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{s \in S} c_{is} x_{is} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   For each store $s \in S$, the total quantity received from all suppliers must equal the demand of the corresponding customer.  
   (Assuming mapping: Customer_1 $\rightarrow$ CLARINDA, Customer_2 $\rightarrow$ FORT MADISON, Customer_3 $\rightarrow$ SIOUX CITY, Customer_4 $\rightarrow$ TOLEDO, Customer_5 $\rightarrow$ BANCROFT)

   \[
   \sum_{i \in I} x_{i,s} = d_j \quad \forall s \in S, \text{ where } j \text{ is the customer for store } s
   \]

   Explicitly:
   - $\sum_{i \in I} x_{i,\text{CLARINDA}} = 2397$
   - $\sum_{i \in I} x_{i,\text{FORT MADISON}} = 1889$
   - $\sum_{i \in I} x_{i,\text{SIOUX CITY}} = 2518$
   - $\sum_{i \in I} x_{i,\text{TOLEDO}} = 3218$
   - $\sum_{i \in I} x_{i,\text{BANCROFT}} = 1813$

2. **Supplier activation constraint:**  
   A supplier can only ship if it is activated. Let $M = \sum_{j} d_j = 11835$ (a valid upper bound):

   \[
   \sum_{s \in S} x_{i,s} \leq M y_i \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{i,s} \geq 0 \quad \forall i \in I,\, s \in S
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{s \in S} c_{is} x_{is} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{i,\text{CLARINDA}} = 2397 \\
& \sum_{i \in I} x_{i,\text{FORT MADISON}} = 1889 \\
& \sum_{i \in I} x_{i,\text{SIOUX CITY}} = 2518 \\
& \sum_{i \in I} x_{i,\text{TOLEDO}} = 3218 \\
& \sum_{i \in I} x_{i,\text{BANCROFT}} = 1813 \\
& \sum_{s \in S} x_{i,s} \leq 11835\, y_i \quad \forall i \in I \\
& x_{i,s} \geq 0 \quad \forall i \in I,\, s \in S \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

##### Parameters (explicit):

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- $d = [2397, 1889, 2518, 3218, 1813]$
- $f = [96.58, 94.06, 94.37, 82.88, 94.96]$
- $C =$
  \[
  \begin{array}{l|ccccc}
  & \text{CLARINDA} & \text{FORT MADISON} & \text{SIOUX CITY} & \text{TOLEDO} & \text{BANCROFT} \\
  \hline
  \text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
  \text{WAUKEE} & 15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
  \text{WAVERLY} & 2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
  \text{PELLA} & 1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
  \text{DES MOINES} & 1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
  \end{array}
  \]
- $M = 11835$