##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- Facilities (Suppliers): $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- Customers (Stores): $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$

- Demand for each customer (from demand.csv, current period):
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Fixed cost for each facility (from fixed_cost.csv, current period):
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation cost per unit from each facility to each customer (from transportation_costs.csv, current period):

|                | Customer_1 (SIOUX CITY) | Customer_2 (CLARINDA) | Customer_3 (FORT MADISON) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|----------------|------------------------|-----------------------|---------------------------|---------------------|-----------------------|
| MOUNT AYR      | 20.07                  | 694.68                | 17.48                     | 199.02              | 1685.53               |
| WAUKEE         | 1.43                   | 15.13                 | 1.5                       | 27.88               | 90.69                 |
| WAVERLY        | 246.6                  | 2.34                  | 349.34                    | 41.3                | 78.73                 |
| PELLA          | 1646.36                | 1181.6                | 1458.53                   | 1924.55             | 38.93                 |
| DES MOINES     | 932.43                 | 1030.8                | 43.48                     | 55.39               | 103.84                |

Let $c_{ij}$ denote the transportation cost per unit from facility $i$ to customer $j$ as above.

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for total shipments from any facility, since there are no explicit capacity limits).

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

2. **Facility activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (If $y_i = 0$, then $x_{ij} = 0$ for all $j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model (with all parameters)

Let
- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$
- $d_{\text{Customer\_1}} = 2397$, $d_{\text{Customer\_2}} = 1889$, $d_{\text{Customer\_3}} = 2518$, $d_{\text{Customer\_4}} = 3218$, $d_{\text{Customer\_5}} = 1813$
- $f_{\text{MOUNT AYR}} = 96.58$, $f_{\text{WAUKEE}} = 94.06$, $f_{\text{WAVERLY}} = 94.37$, $f_{\text{PELLA}} = 82.88$, $f_{\text{DES MOINES}} = 94.96$
- $c_{ij}$ as in the table above
- $M = 11835$

**Variables:**
- $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$ (binary)

**Objective:**
\[
\min \left(
\begin{aligned}
&20.07\,x_{\text{MOUNT AYR},\,\text{Customer\_1}} + 694.68\,x_{\text{MOUNT AYR},\,\text{Customer\_2}} + 17.48\,x_{\text{MOUNT AYR},\,\text{Customer\_3}} + 199.02\,x_{\text{MOUNT AYR},\,\text{Customer\_4}} + 1685.53\,x_{\text{MOUNT AYR},\,\text{Customer\_5}} \\
+&1.43\,x_{\text{WAUKEE},\,\text{Customer\_1}} + 15.13\,x_{\text{WAUKEE},\,\text{Customer\_2}} + 1.5\,x_{\text{WAUKEE},\,\text{Customer\_3}} + 27.88\,x_{\text{WAUKEE},\,\text{Customer\_4}} + 90.69\,x_{\text{WAUKEE},\,\text{Customer\_5}} \\
+&246.6\,x_{\text{WAVERLY},\,\text{Customer\_1}} + 2.34\,x_{\text{WAVERLY},\,\text{Customer\_2}} + 349.34\,x_{\text{WAVERLY},\,\text{Customer\_3}} + 41.3\,x_{\text{WAVERLY},\,\text{Customer\_4}} + 78.73\,x_{\text{WAVERLY},\,\text{Customer\_5}} \\
+&1646.36\,x_{\text{PELLA},\,\text{Customer\_1}} + 1181.6\,x_{\text{PELLA},\,\text{Customer\_2}} + 1458.53\,x_{\text{PELLA},\,\text{Customer\_3}} + 1924.55\,x_{\text{PELLA},\,\text{Customer\_4}} + 38.93\,x_{\text{PELLA},\,\text{Customer\_5}} \\
+&932.43\,x_{\text{DES MOINES},\,\text{Customer\_1}} + 1030.8\,x_{\text{DES MOINES},\,\text{Customer\_2}} + 43.48\,x_{\text{DES MOINES},\,\text{Customer\_3}} + 55.39\,x_{\text{DES MOINES},\,\text{Customer\_4}} + 103.84\,x_{\text{DES MOINES},\,\text{Customer\_5}} \\
+&96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
\end{aligned}
\right)
\]

**Subject to:**

For each customer:
\[
\begin{aligned}
x_{\text{MOUNT AYR},\,\text{Customer\_1}} + x_{\text{WAUKEE},\,\text{Customer\_1}} + x_{\text{WAVERLY},\,\text{Customer\_1}} + x_{\text{PELLA},\,\text{Customer\_1}} + x_{\text{DES MOINES},\,\text{Customer\_1}} &= 2397 \\
x_{\text{MOUNT AYR},\,\text{Customer\_2}} + x_{\text{WAUKEE},\,\text{Customer\_2}} + x_{\text{WAVERLY},\,\text{Customer\_2}} + x_{\text{PELLA},\,\text{Customer\_2}} + x_{\text{DES MOINES},\,\text{Customer\_2}} &= 1889 \\
x_{\text{MOUNT AYR},\,\text{Customer\_3}} + x_{\text{WAUKEE},\,\text{Customer\_3}} + x_{\text{WAVERLY},\,\text{Customer\_3}} + x_{\text{PELLA},\,\text{Customer\_3}} + x_{\text{DES MOINES},\,\text{Customer\_3}} &= 2518 \\
x_{\text{MOUNT AYR},\,\text{Customer\_4}} + x_{\text{WAUKEE},\,\text{Customer\_4}} + x_{\text{WAVERLY},\,\text{Customer\_4}} + x_{\text{PELLA},\,\text{Customer\_4}} + x_{\text{DES MOINES},\,\text{Customer\_4}} &= 3218 \\
x_{\text{MOUNT AYR},\,\text{Customer\_5}} + x_{\text{WAUKEE},\,\text{Customer\_5}} + x_{\text{WAVERLY},\,\text{Customer\_5}} + x_{\text{PELLA},\,\text{Customer\_5}} + x_{\text{DES MOINES},\,\text{Customer\_5}} &= 1813 \\
\end{aligned}
\]

For each facility:
\[
\begin{aligned}
x_{\text{MOUNT AYR},\,\text{Customer\_1}} + x_{\text{MOUNT AYR},\,\text{Customer\_2}} + x_{\text{MOUNT AYR},\,\text{Customer\_3}} + x_{\text{MOUNT AYR},\,\text{Customer\_4}} + x_{\text{MOUNT AYR},\,\text{Customer\_5}} &\leq 11835\,y_{\text{MOUNT AYR}} \\
x_{\text{WAUKEE},\,\text{Customer\_1}} + x_{\text{WAUKEE},\,\text{Customer\_2}} + x_{\text{WAUKEE},\,\text{Customer\_3}} + x_{\text{WAUKEE},\,\text{Customer\_4}} + x_{\text{WAUKEE},\,\text{Customer\_5}} &\leq 11835\,y_{\text{WAUKEE}} \\
x_{\text{WAVERLY},\,\text{Customer\_1}} + x_{\text{WAVERLY},\,\text{Customer\_2}} + x_{\text{WAVERLY},\,\text{Customer\_3}} + x_{\text{WAVERLY},\,\text{Customer\_4}} + x_{\text{WAVERLY},\,\text{Customer\_5}} &\leq 11835\,y_{\text{WAVERLY}} \\
x_{\text{PELLA},\,\text{Customer\_1}} + x_{\text{PELLA},\,\text{Customer\_2}} + x_{\text{PELLA},\,\text{Customer\_3}} + x_{\text{PELLA},\,\text{Customer\_4}} + x_{\text{PELLA},\,\text{Customer\_5}} &\leq 11835\,y_{\text{PELLA}} \\
x_{\text{DES MOINES},\,\text{Customer\_1}} + x_{\text{DES MOINES},\,\text{Customer\_2}} + x_{\text{DES MOINES},\,\text{Customer\_3}} + x_{\text{DES MOINES},\,\text{Customer\_4}} + x_{\text{DES MOINES},\,\text{Customer\_5}} &\leq 11835\,y_{\text{DES MOINES}} \\
\end{aligned}
\]

And
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

---

**All parameters, indices, and coefficients are preserved exactly as in the source data.**