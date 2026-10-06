##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$.
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (operational), 0 otherwise.

##### Parameters

- Facilities (Suppliers): $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Customers (Stores): $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5},\ \text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

- Fixed costs for each facility $i$:
  - $\text{MOUNT AYR}$: $f_{\text{MOUNT AYR}} = 96.58$
  - $\text{WAUKEE}$: $f_{\text{WAUKEE}} = 94.06$
  - $\text{WAVERLY}$: $f_{\text{WAVERLY}} = 94.37$
  - $\text{PELLA}$: $f_{\text{PELLA}} = 82.88$
  - $\text{DES MOINES}$: $f_{\text{DES MOINES}} = 94.96$

- Demand for each customer $j$:
  - $\text{Customer\_1}$: $d_{\text{Customer\_1}} = 2397$
  - $\text{Customer\_2}$: $d_{\text{Customer\_2}} = 1889$
  - $\text{Customer\_3}$: $d_{\text{Customer\_3}} = 2518$
  - $\text{Customer\_4}$: $d_{\text{Customer\_4}} = 3218$
  - $\text{Customer\_5}$: $d_{\text{Customer\_5}} = 1813$
  - (No explicit demand provided for CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT.)

- Transportation costs $c_{ij}$ (from facility $i$ to customer $j$):

| Facility      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where $c_{ij}$ is the transportation cost per unit from facility $i$ to customer $j$, and $f_i$ is the fixed cost for opening facility $i$.

##### Constraints

1. **Demand Satisfaction:**  
   For each customer $j$ with specified demand,
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J \text{ with specified } d_j
   \]
   (For customers without specified demand, no constraint is imposed.)

2. **Facility Activation:**  
   For each facility $i$,
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i \qquad \forall i \in I
   \]
   where $M$ is a sufficiently large constant, e.g., $M = \sum_{j \in J} d_j = 11835$ (sum of all specified demands).

3. **Nonnegativity and Binary Variables:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

##### Complete Parameter Listing

- Facilities (Suppliers):  
  $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$

- Customers (Stores):  
  $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5},\ \text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

- Demands:  
  $d_{\text{Customer\_1}} = 2397$  
  $d_{\text{Customer\_2}} = 1889$  
  $d_{\text{Customer\_3}} = 2518$  
  $d_{\text{Customer\_4}} = 3218$  
  $d_{\text{Customer\_5}} = 1813$  
  (No demand specified for CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT.)

- Fixed Costs:  
  $f_{\text{MOUNT AYR}} = 96.58$  
  $f_{\text{WAUKEE}} = 94.06$  
  $f_{\text{WAVERLY}} = 94.37$  
  $f_{\text{PELLA}} = 82.88$  
  $f_{\text{DES MOINES}} = 94.96$

- Transportation Costs $c_{ij}$:  
  As in the table above.

- $M = 11835$ (sum of all specified demands).

---

**Mathematical Model:**

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\} \\
& \sum_{j \in J} x_{ij} \leq M y_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I
\end{align*}
\]

with all parameters and data as listed above.