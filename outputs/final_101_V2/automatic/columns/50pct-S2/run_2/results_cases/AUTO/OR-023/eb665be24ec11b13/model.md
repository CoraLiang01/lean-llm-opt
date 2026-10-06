##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i$ to store (customer) $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- Facilities (Suppliers): $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Customers (Stores): $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$

- Fixed costs $f_i$ (per facility):

  - MOUNT AYR: $96.58$
  - WAUKEE: $94.06$
  - WAVERLY: $94.37$
  - PELLA: $82.88$
  - DES MOINES: $94.96$

- Transportation costs $c_{ij}$ (per unit from facility $i$ to customer $j$):

  |                | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
  |----------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
  | MOUNT AYR      | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
  | WAUKEE         | 15.13                 | 1.5                       | 1.43                    | 27.88               | 90.69                 |
  | WAVERLY        | 2.34                  | 349.34                    | 246.6                   | 41.3                | 78.73                 |
  | PELLA          | 1181.6                | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
  | DES MOINES     | 1030.8                | 43.48                     | 932.43                  | 55.39               | 103.84                |

- Demand $d_j$ (per customer):

  - Customer_1: $2397$
  - Customer_2: $1889$
  - Customer_3: $2518$
  - Customer_4: $3218$
  - Customer_5: $1813$

- Let $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$ (sufficiently large upper bound for linking $x_{ij}$ and $y_i$).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (no shipments from closed suppliers):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model (with all parameters)

Let $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$, $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$, $M = 11835$.

\[
\begin{align*}
\min\ & \Big[694.68\,x_{\text{MOUNT AYR},\text{Customer\_1}} + 17.48\,x_{\text{MOUNT AYR},\text{Customer\_2}} + 20.07\,x_{\text{MOUNT AYR},\text{Customer\_3}} + 199.02\,x_{\text{MOUNT AYR},\text{Customer\_4}} + 1685.53\,x_{\text{MOUNT AYR},\text{Customer\_5}} \\
&+ 15.13\,x_{\text{WAUKEE},\text{Customer\_1}} + 1.5\,x_{\text{WAUKEE},\text{Customer\_2}} + 1.43\,x_{\text{WAUKEE},\text{Customer\_3}} + 27.88\,x_{\text{WAUKEE},\text{Customer\_4}} + 90.69\,x_{\text{WAUKEE},\text{Customer\_5}} \\
&+ 2.34\,x_{\text{WAVERLY},\text{Customer\_1}} + 349.34\,x_{\text{WAVERLY},\text{Customer\_2}} + 246.6\,x_{\text{WAVERLY},\text{Customer\_3}} + 41.3\,x_{\text{WAVERLY},\text{Customer\_4}} + 78.73\,x_{\text{WAVERLY},\text{Customer\_5}} \\
&+ 1181.6\,x_{\text{PELLA},\text{Customer\_1}} + 1458.53\,x_{\text{PELLA},\text{Customer\_2}} + 1646.36\,x_{\text{PELLA},\text{Customer\_3}} + 1924.55\,x_{\text{PELLA},\text{Customer\_4}} + 38.93\,x_{\text{PELLA},\text{Customer\_5}} \\
&+ 1030.8\,x_{\text{DES MOINES},\text{Customer\_1}} + 43.48\,x_{\text{DES MOINES},\text{Customer\_2}} + 932.43\,x_{\text{DES MOINES},\text{Customer\_3}} + 55.39\,x_{\text{DES MOINES},\text{Customer\_4}} + 103.84\,x_{\text{DES MOINES},\text{Customer\_5}} \\
&+ 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}} \Big]
\end{align*}
\]

Subject to:

\[
\begin{align*}
&\sum_{i \in I} x_{i,\text{Customer\_1}} = 2397 \\
&\sum_{i \in I} x_{i,\text{Customer\_2}} = 1889 \\
&\sum_{i \in I} x_{i,\text{Customer\_3}} = 2518 \\
&\sum_{i \in I} x_{i,\text{Customer\_4}} = 3218 \\
&\sum_{i \in I} x_{i,\text{Customer\_5}} = 1813 \\
\\
&\sum_{j \in J} x_{\text{MOUNT AYR},j} \leq 11835\,y_{\text{MOUNT AYR}} \\
&\sum_{j \in J} x_{\text{WAUKEE},j} \leq 11835\,y_{\text{WAUKEE}} \\
&\sum_{j \in J} x_{\text{WAVERLY},j} \leq 11835\,y_{\text{WAVERLY}} \\
&\sum_{j \in J} x_{\text{PELLA},j} \leq 11835\,y_{\text{PELLA}} \\
&\sum_{j \in J} x_{\text{DES MOINES},j} \leq 11835\,y_{\text{DES MOINES}} \\
\\
&x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
&y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

All parameters, vectors, and matrices are included as retrieved. This model determines which suppliers to activate and how to allocate shipments to minimize total cost while meeting all store demands.