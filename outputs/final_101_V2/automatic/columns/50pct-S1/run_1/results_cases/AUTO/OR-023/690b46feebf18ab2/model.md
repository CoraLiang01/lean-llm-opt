##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- Facilities (Suppliers): $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Customers (Stores): $J = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$

- Fixed costs $f_i$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demands $d_j$:
  - $d_{\text{CLARINDA}} = 2397$
  - $d_{\text{FORT MADISON}} = 1889$
  - $d_{\text{SIOUX CITY}} = 2518$
  - $d_{\text{TOLEDO}} = 3218$
  - $d_{\text{BANCROFT}} = 1813$

- Transportation costs $c_{ij}$:

|                | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE         | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY        | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA          | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES     | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for total shipments from any supplier).

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
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
\min\ & \Bigg[
\ 694.68\,x_{\text{MOUNT AYR},\,\text{CLARINDA}} + 17.48\,x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + 20.07\,x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + 199.02\,x_{\text{MOUNT AYR},\,\text{TOLEDO}} + 1685.53\,x_{\text{MOUNT AYR},\,\text{BANCROFT}} \\
&+ 15.13\,x_{\text{WAUKEE},\,\text{CLARINDA}} + 1.50\,x_{\text{WAUKEE},\,\text{FORT MADISON}} + 1.43\,x_{\text{WAUKEE},\,\text{SIOUX CITY}} + 27.88\,x_{\text{WAUKEE},\,\text{TOLEDO}} + 90.69\,x_{\text{WAUKEE},\,\text{BANCROFT}} \\
&+ 2.34\,x_{\text{WAVERLY},\,\text{CLARINDA}} + 349.34\,x_{\text{WAVERLY},\,\text{FORT MADISON}} + 246.60\,x_{\text{WAVERLY},\,\text{SIOUX CITY}} + 41.30\,x_{\text{WAVERLY},\,\text{TOLEDO}} + 78.73\,x_{\text{WAVERLY},\,\text{BANCROFT}} \\
&+ 1181.60\,x_{\text{PELLA},\,\text{CLARINDA}} + 1458.53\,x_{\text{PELLA},\,\text{FORT MADISON}} + 1646.36\,x_{\text{PELLA},\,\text{SIOUX CITY}} + 1924.55\,x_{\text{PELLA},\,\text{TOLEDO}} + 38.93\,x_{\text{PELLA},\,\text{BANCROFT}} \\
&+ 1030.80\,x_{\text{DES MOINES},\,\text{CLARINDA}} + 43.48\,x_{\text{DES MOINES},\,\text{FORT MADISON}} + 932.43\,x_{\text{DES MOINES},\,\text{SIOUX CITY}} + 55.39\,x_{\text{DES MOINES},\,\text{TOLEDO}} + 103.84\,x_{\text{DES MOINES},\,\text{BANCROFT}} \\
&+ 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
\ \Bigg]
\end{align*}
\]

Subject to:

For each customer:
- $\sum_{i \in I} x_{ij} = d_j$

\[
\begin{align*}
x_{\text{MOUNT AYR},\,\text{CLARINDA}} + x_{\text{WAUKEE},\,\text{CLARINDA}} + x_{\text{WAVERLY},\,\text{CLARINDA}} + x_{\text{PELLA},\,\text{CLARINDA}} + x_{\text{DES MOINES},\,\text{CLARINDA}} &= 2397 \\
x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + x_{\text{WAUKEE},\,\text{FORT MADISON}} + x_{\text{WAVERLY},\,\text{FORT MADISON}} + x_{\text{PELLA},\,\text{FORT MADISON}} + x_{\text{DES MOINES},\,\text{FORT MADISON}} &= 1889 \\
x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + x_{\text{WAUKEE},\,\text{SIOUX CITY}} + x_{\text{WAVERLY},\,\text{SIOUX CITY}} + x_{\text{PELLA},\,\text{SIOUX CITY}} + x_{\text{DES MOINES},\,\text{SIOUX CITY}} &= 2518 \\
x_{\text{MOUNT AYR},\,\text{TOLEDO}} + x_{\text{WAUKEE},\,\text{TOLEDO}} + x_{\text{WAVERLY},\,\text{TOLEDO}} + x_{\text{PELLA},\,\text{TOLEDO}} + x_{\text{DES MOINES},\,\text{TOLEDO}} &= 3218 \\
x_{\text{MOUNT AYR},\,\text{BANCROFT}} + x_{\text{WAUKEE},\,\text{BANCROFT}} + x_{\text{WAVERLY},\,\text{BANCROFT}} + x_{\text{PELLA},\,\text{BANCROFT}} + x_{\text{DES MOINES},\,\text{BANCROFT}} &= 1813 \\
\end{align*}
\]

For each supplier:
- $\sum_{j \in J} x_{ij} \leq 11835\, y_i$

\[
\begin{align*}
x_{\text{MOUNT AYR},\,\text{CLARINDA}} + x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + x_{\text{MOUNT AYR},\,\text{TOLEDO}} + x_{\text{MOUNT AYR},\,\text{BANCROFT}} &\leq 11835\, y_{\text{MOUNT AYR}} \\
x_{\text{WAUKEE},\,\text{CLARINDA}} + x_{\text{WAUKEE},\,\text{FORT MADISON}} + x_{\text{WAUKEE},\,\text{SIOUX CITY}} + x_{\text{WAUKEE},\,\text{TOLEDO}} + x_{\text{WAUKEE},\,\text{BANCROFT}} &\leq 11835\, y_{\text{WAUKEE}} \\
x_{\text{WAVERLY},\,\text{CLARINDA}} + x_{\text{WAVERLY},\,\text{FORT MADISON}} + x_{\text{WAVERLY},\,\text{SIOUX CITY}} + x_{\text{WAVERLY},\,\text{TOLEDO}} + x_{\text{WAVERLY},\,\text{BANCROFT}} &\leq 11835\, y_{\text{WAVERLY}} \\
x_{\text{PELLA},\,\text{CLARINDA}} + x_{\text{PELLA},\,\text{FORT MADISON}} + x_{\text{PELLA},\,\text{SIOUX CITY}} + x_{\text{PELLA},\,\text{TOLEDO}} + x_{\text{PELLA},\,\text{BANCROFT}} &\leq 11835\, y_{\text{PELLA}} \\
x_{\text{DES MOINES},\,\text{CLARINDA}} + x_{\text{DES MOINES},\,\text{FORT MADISON}} + x_{\text{DES MOINES},\,\text{SIOUX CITY}} + x_{\text{DES MOINES},\,\text{TOLEDO}} + x_{\text{DES MOINES},\,\text{BANCROFT}} &\leq 11835\, y_{\text{DES MOINES}} \\
\end{align*}
\]

And for all $i \in I$, $j \in J$:
- $x_{ij} \geq 0$
- $y_i \in \{0,1\}$

---

**All parameters, vectors, and matrices are explicitly included as retrieved from the CSV files.**