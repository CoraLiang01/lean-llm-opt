##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (operational), 0 otherwise.

##### Sets and Indices

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- Customers (for demand): $\{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$

##### Parameters

- Demand at each store (in order):
  - $d_1 = 2397$ (Customer_1)
  - $d_2 = 1889$ (Customer_2)
  - $d_3 = 2518$ (Customer_3)
  - $d_4 = 3218$ (Customer_4)
  - $d_5 = 1813$ (Customer_5)

- Fixed costs for each supplier:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs $c_{ij}$ (supplier $i$, store $j$):

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA         | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

- Let $M = \sum_{j=1}^5 d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (A supplier can only ship if it is activated.)

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Explicit Model with Parameters

Let $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$, $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$, and $d_j$ as above.

\[
\begin{align*}
\min \quad & 
\Big[
694.68\,x_{\text{MOUNT AYR},\text{CLARINDA}} + 17.48\,x_{\text{MOUNT AYR},\text{FORT MADISON}} + 20.07\,x_{\text{MOUNT AYR},\text{SIOUX CITY}} + 199.02\,x_{\text{MOUNT AYR},\text{TOLEDO}} + 1685.53\,x_{\text{MOUNT AYR},\text{BANCROFT}} \\
&+ 15.13\,x_{\text{WAUKEE},\text{CLARINDA}} + 1.50\,x_{\text{WAUKEE},\text{FORT MADISON}} + 1.43\,x_{\text{WAUKEE},\text{SIOUX CITY}} + 27.88\,x_{\text{WAUKEE},\text{TOLEDO}} + 90.69\,x_{\text{WAUKEE},\text{BANCROFT}} \\
&+ 2.34\,x_{\text{WAVERLY},\text{CLARINDA}} + 349.34\,x_{\text{WAVERLY},\text{FORT MADISON}} + 246.60\,x_{\text{WAVERLY},\text{SIOUX CITY}} + 41.30\,x_{\text{WAVERLY},\text{TOLEDO}} + 78.73\,x_{\text{WAVERLY},\text{BANCROFT}} \\
&+ 1181.60\,x_{\text{PELLA},\text{CLARINDA}} + 1458.53\,x_{\text{PELLA},\text{FORT MADISON}} + 1646.36\,x_{\text{PELLA},\text{SIOUX CITY}} + 1924.55\,x_{\text{PELLA},\text{TOLEDO}} + 38.93\,x_{\text{PELLA},\text{BANCROFT}} \\
&+ 1030.80\,x_{\text{DES MOINES},\text{CLARINDA}} + 43.48\,x_{\text{DES MOINES},\text{FORT MADISON}} + 932.43\,x_{\text{DES MOINES},\text{SIOUX CITY}} + 55.39\,x_{\text{DES MOINES},\text{TOLEDO}} + 103.84\,x_{\text{DES MOINES},\text{BANCROFT}} \\
&+ 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
\Big]
\end{align*}
\]

Subject to:

For each store (in order: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT):

\[
\begin{align*}
x_{\text{MOUNT AYR},j} + x_{\text{WAUKEE},j} + x_{\text{WAVERLY},j} + x_{\text{PELLA},j} + x_{\text{DES MOINES},j} = d_j, \quad \forall j \in J
\end{align*}
\]

For each supplier:

\[
\begin{align*}
x_{i,\text{CLARINDA}} + x_{i,\text{FORT MADISON}} + x_{i,\text{SIOUX CITY}} + x_{i,\text{TOLEDO}} + x_{i,\text{BANCROFT}} \leq 11835\,y_i, \quad \forall i \in I
\end{align*}
\]

\[
x_{ij} \geq 0, \quad y_i \in \{0,1\}
\]

##### Retrieved Information

- Suppliers: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Stores: CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT
- Demands: 2397, 1889, 2518, 3218, 1813 (in order)
- Fixed costs: 96.58, 94.06, 94.37, 82.88, 94.96 (in order)
- Transportation costs: as in the table above
- $M = 11835$