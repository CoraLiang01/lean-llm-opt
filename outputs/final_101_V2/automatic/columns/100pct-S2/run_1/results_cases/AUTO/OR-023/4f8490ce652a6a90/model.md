##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$.
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (operational), 0 otherwise.

##### Parameters

- Let $I$ be the set of suppliers (facilities):  
  $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$

- Let $J$ be the set of stores (customers):  
  $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$

- Fixed costs for each supplier:
  \[
  \begin{align*}
  f_{\text{MOUNT AYR}} &= 96.58 \\
  f_{\text{WAUKEE}} &= 94.06 \\
  f_{\text{WAVERLY}} &= 94.37 \\
  f_{\text{PELLA}} &= 82.88 \\
  f_{\text{DES MOINES}} &= 94.96 \\
  \end{align*}
  \]

- Demand for each store:
  \[
  \begin{align*}
  d_{\text{Customer\_1}} &= 2397 \\
  d_{\text{Customer\_2}} &= 1889 \\
  d_{\text{Customer\_3}} &= 2518 \\
  d_{\text{Customer\_4}} &= 3218 \\
  d_{\text{Customer\_5}} &= 1813 \\
  \end{align*}
  \]

- Transportation costs per unit from each supplier to each store (as per the original matrix, mapping preserved):

  | Supplier      | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
  |---------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
  | MOUNT AYR     | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
  | WAUKEE        | 15.13                 | 1.50                      | 1.43                    | 27.88               | 90.69                 |
  | WAVERLY       | 2.34                  | 349.34                    | 246.60                  | 41.30               | 78.73                 |
  | PELLA         | 1181.60               | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
  | DES MOINES    | 1030.80               | 43.48                     | 932.43                  | 55.39               | 103.84                |

  Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to customer $j$ as above.

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (a valid upper bound for any supplier's total shipment, since there are no explicit supplier capacity limits).

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
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
\ 694.68\,x_{\text{MOUNT AYR},\,\text{Customer\_1}} + 17.48\,x_{\text{MOUNT AYR},\,\text{Customer\_2}} + 20.07\,x_{\text{MOUNT AYR},\,\text{Customer\_3}} + 199.02\,x_{\text{MOUNT AYR},\,\text{Customer\_4}} + 1685.53\,x_{\text{MOUNT AYR},\,\text{Customer\_5}} \\
&+ 15.13\,x_{\text{WAUKEE},\,\text{Customer\_1}} + 1.50\,x_{\text{WAUKEE},\,\text{Customer\_2}} + 1.43\,x_{\text{WAUKEE},\,\text{Customer\_3}} + 27.88\,x_{\text{WAUKEE},\,\text{Customer\_4}} + 90.69\,x_{\text{WAUKEE},\,\text{Customer\_5}} \\
&+ 2.34\,x_{\text{WAVERLY},\,\text{Customer\_1}} + 349.34\,x_{\text{WAVERLY},\,\text{Customer\_2}} + 246.60\,x_{\text{WAVERLY},\,\text{Customer\_3}} + 41.30\,x_{\text{WAVERLY},\,\text{Customer\_4}} + 78.73\,x_{\text{WAVERLY},\,\text{Customer\_5}} \\
&+ 1181.60\,x_{\text{PELLA},\,\text{Customer\_1}} + 1458.53\,x_{\text{PELLA},\,\text{Customer\_2}} + 1646.36\,x_{\text{PELLA},\,\text{Customer\_3}} + 1924.55\,x_{\text{PELLA},\,\text{Customer\_4}} + 38.93\,x_{\text{PELLA},\,\text{Customer\_5}} \\
&+ 1030.80\,x_{\text{DES MOINES},\,\text{Customer\_1}} + 43.48\,x_{\text{DES MOINES},\,\text{Customer\_2}} + 932.43\,x_{\text{DES MOINES},\,\text{Customer\_3}} + 55.39\,x_{\text{DES MOINES},\,\text{Customer\_4}} + 103.84\,x_{\text{DES MOINES},\,\text{Customer\_5}} \\
&+ 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
\ \Bigg]
\end{align*}
\]

Subject to:

For each customer:
\[
\begin{align*}
x_{\text{MOUNT AYR},\,\text{Customer\_1}} + x_{\text{WAUKEE},\,\text{Customer\_1}} + x_{\text{WAVERLY},\,\text{Customer\_1}} + x_{\text{PELLA},\,\text{Customer\_1}} + x_{\text{DES MOINES},\,\text{Customer\_1}} &= 2397 \\
x_{\text{MOUNT AYR},\,\text{Customer\_2}} + x_{\text{WAUKEE},\,\text{Customer\_2}} + x_{\text{WAVERLY},\,\text{Customer\_2}} + x_{\text{PELLA},\,\text{Customer\_2}} + x_{\text{DES MOINES},\,\text{Customer\_2}} &= 1889 \\
x_{\text{MOUNT AYR},\,\text{Customer\_3}} + x_{\text{WAUKEE},\,\text{Customer\_3}} + x_{\text{WAVERLY},\,\text{Customer\_3}} + x_{\text{PELLA},\,\text{Customer\_3}} + x_{\text{DES MOINES},\,\text{Customer\_3}} &= 2518 \\
x_{\text{MOUNT AYR},\,\text{Customer\_4}} + x_{\text{WAUKEE},\,\text{Customer\_4}} + x_{\text{WAVERLY},\,\text{Customer\_4}} + x_{\text{PELLA},\,\text{Customer\_4}} + x_{\text{DES MOINES},\,\text{Customer\_4}} &= 3218 \\
x_{\text{MOUNT AYR},\,\text{Customer\_5}} + x_{\text{WAUKEE},\,\text{Customer\_5}} + x_{\text{WAVERLY},\,\text{Customer\_5}} + x_{\text{PELLA},\,\text{Customer\_5}} + x_{\text{DES MOINES},\,\text{Customer\_5}} &= 1813 \\
\end{align*}
\]

For each supplier:
\[
\begin{align*}
x_{\text{MOUNT AYR},\,\text{Customer\_1}} + x_{\text{MOUNT AYR},\,\text{Customer\_2}} + x_{\text{MOUNT AYR},\,\text{Customer\_3}} + x_{\text{MOUNT AYR},\,\text{Customer\_4}} + x_{\text{MOUNT AYR},\,\text{Customer\_5}} &\leq 11835\,y_{\text{MOUNT AYR}} \\
x_{\text{WAUKEE},\,\text{Customer\_1}} + x_{\text{WAUKEE},\,\text{Customer\_2}} + x_{\text{WAUKEE},\,\text{Customer\_3}} + x_{\text{WAUKEE},\,\text{Customer\_4}} + x_{\text{WAUKEE},\,\text{Customer\_5}} &\leq 11835\,y_{\text{WAUKEE}} \\
x_{\text{WAVERLY},\,\text{Customer\_1}} + x_{\text{WAVERLY},\,\text{Customer\_2}} + x_{\text{WAVERLY},\,\text{Customer\_3}} + x_{\text{WAVERLY},\,\text{Customer\_4}} + x_{\text{WAVERLY},\,\text{Customer\_5}} &\leq 11835\,y_{\text{WAVERLY}} \\
x_{\text{PELLA},\,\text{Customer\_1}} + x_{\text{PELLA},\,\text{Customer\_2}} + x_{\text{PELLA},\,\text{Customer\_3}} + x_{\text{PELLA},\,\text{Customer\_4}} + x_{\text{PELLA},\,\text{Customer\_5}} &\leq 11835\,y_{\text{PELLA}} \\
x_{\text{DES MOINES},\,\text{Customer\_1}} + x_{\text{DES MOINES},\,\text{Customer\_2}} + x_{\text{DES MOINES},\,\text{Customer\_3}} + x_{\text{DES MOINES},\,\text{Customer\_4}} + x_{\text{DES MOINES},\,\text{Customer\_5}} &\leq 11835\,y_{\text{DES MOINES}} \\
\end{align*}
\]

And for all $i \in I$, $j \in J$:
\[
x_{ij} \geq 0,\quad y_i \in \{0,1\}
\]

---

**All parameters, vectors, and matrices are explicitly included as retrieved. The model above fully represents the Iowa Department of Commerce's supplier activation and shipment minimization problem as described.**