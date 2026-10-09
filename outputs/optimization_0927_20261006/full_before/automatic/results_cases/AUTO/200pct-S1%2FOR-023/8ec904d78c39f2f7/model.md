##### Decision Variables

Let:
- $x_{ij} \geq 0$: quantity of goods shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

Where:
- $i \in I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $j \in J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$

##### Parameters

- Fixed costs for each supplier:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demand for each store:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Transportation costs $c_{ij}$ (per unit from supplier $i$ to store $j$):

| Supplier \ Store | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|------------------|----------|--------------|------------|--------|----------|
| MOUNT AYR        | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE           | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY          | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA            | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES       | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

##### Objective Function

\[
\min \left(
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
+ \sum_{i \in I} f_i y_i
\right)
\]

That is,

\[
\min \Bigg[
\begin{aligned}
&694.68\,x_{\text{MOUNT AYR},\,\text{CLARINDA}} + 17.48\,x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + 20.07\,x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + 199.02\,x_{\text{MOUNT AYR},\,\text{TOLEDO}} + 1685.53\,x_{\text{MOUNT AYR},\,\text{BANCROFT}} \\
+& 15.13\,x_{\text{WAUKEE},\,\text{CLARINDA}} + 1.5\,x_{\text{WAUKEE},\,\text{FORT MADISON}} + 1.43\,x_{\text{WAUKEE},\,\text{SIOUX CITY}} + 27.88\,x_{\text{WAUKEE},\,\text{TOLEDO}} + 90.69\,x_{\text{WAUKEE},\,\text{BANCROFT}} \\
+& 2.34\,x_{\text{WAVERLY},\,\text{CLARINDA}} + 349.34\,x_{\text{WAVERLY},\,\text{FORT MADISON}} + 246.6\,x_{\text{WAVERLY},\,\text{SIOUX CITY}} + 41.3\,x_{\text{WAVERLY},\,\text{TOLEDO}} + 78.73\,x_{\text{WAVERLY},\,\text{BANCROFT}} \\
+& 1181.6\,x_{\text{PELLA},\,\text{CLARINDA}} + 1458.53\,x_{\text{PELLA},\,\text{FORT MADISON}} + 1646.36\,x_{\text{PELLA},\,\text{SIOUX CITY}} + 1924.55\,x_{\text{PELLA},\,\text{TOLEDO}} + 38.93\,x_{\text{PELLA},\,\text{BANCROFT}} \\
+& 1030.8\,x_{\text{DES MOINES},\,\text{CLARINDA}} + 43.48\,x_{\text{DES MOINES},\,\text{FORT MADISON}} + 932.43\,x_{\text{DES MOINES},\,\text{SIOUX CITY}} + 55.39\,x_{\text{DES MOINES},\,\text{TOLEDO}} + 103.84\,x_{\text{DES MOINES},\,\text{BANCROFT}} \\
+& 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
\end{aligned}
\Bigg]
\]

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met exactly.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   That is,
   - $x_{\text{MOUNT AYR},\,\text{CLARINDA}} + x_{\text{WAUKEE},\,\text{CLARINDA}} + x_{\text{WAVERLY},\,\text{CLARINDA}} + x_{\text{PELLA},\,\text{CLARINDA}} + x_{\text{DES MOINES},\,\text{CLARINDA}} = 2397$
   - $x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + x_{\text{WAUKEE},\,\text{FORT MADISON}} + x_{\text{WAVERLY},\,\text{FORT MADISON}} + x_{\text{PELLA},\,\text{FORT MADISON}} + x_{\text{DES MOINES},\,\text{FORT MADISON}} = 1889$
   - $x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + x_{\text{WAUKEE},\,\text{SIOUX CITY}} + x_{\text{WAVERLY},\,\text{SIOUX CITY}} + x_{\text{PELLA},\,\text{SIOUX CITY}} + x_{\text{DES MOINES},\,\text{SIOUX CITY}} = 2518$
   - $x_{\text{MOUNT AYR},\,\text{TOLEDO}} + x_{\text{WAUKEE},\,\text{TOLEDO}} + x_{\text{WAVERLY},\,\text{TOLEDO}} + x_{\text{PELLA},\,\text{TOLEDO}} + x_{\text{DES MOINES},\,\text{TOLEDO}} = 3218$
   - $x_{\text{MOUNT AYR},\,\text{BANCROFT}} + x_{\text{WAUKEE},\,\text{BANCROFT}} + x_{\text{WAVERLY},\,\text{BANCROFT}} + x_{\text{PELLA},\,\text{BANCROFT}} + x_{\text{DES MOINES},\,\text{BANCROFT}} = 1813$

2. **Facility activation:** No shipments from a supplier unless it is open. Since there are no explicit capacity limits, use a big-M constraint with $M = \sum_{j \in J} d_j = 11835$.
   \[
   \sum_{j \in J} x_{ij} \leq M\,y_i, \quad \forall i \in I
   \]
   That is, for each supplier $i$:
   - $x_{i,\text{CLARINDA}} + x_{i,\text{FORT MADISON}} + x_{i,\text{SIOUX CITY}} + x_{i,\text{TOLEDO}} + x_{i,\text{BANCROFT}} \leq 11835\,y_i$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Summary of Sets and Parameters

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $f_i$ as above
- $d_j$ as above
- $c_{ij}$ as above
- $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

with all parameters and sets as specified above.