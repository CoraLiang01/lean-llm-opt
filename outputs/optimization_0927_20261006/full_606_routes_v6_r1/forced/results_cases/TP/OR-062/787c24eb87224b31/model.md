##### Sets

Let  
$F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$ (suppliers)  
$S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$ (stores/customers)

##### Parameters

Store demands (from demand.csv, in source order):
\[
\begin{align*}
d_{\text{Customer\_1}} &= 2397 \\
d_{\text{Customer\_2}} &= 1889 \\
d_{\text{Customer\_3}} &= 2518 \\
d_{\text{Customer\_4}} &= 3218 \\
d_{\text{Customer\_5}} &= 1813 \\
\end{align*}
\]
Let the mapping from Customer_k to store names be:
- Customer_1 = CLARINDA
- Customer_2 = FORT MADISON
- Customer_3 = SIOUX CITY
- Customer_4 = TOLEDO
- Customer_5 = BANCROFT

Supplier fixed costs (from fixed_cost.csv, in source order):
\[
\begin{align*}
f_{\text{MOUNT AYR}} &= 96.58 \\
f_{\text{WAUKEE}} &= 94.06 \\
f_{\text{WAVERLY}} &= 94.37 \\
f_{\text{PELLA}} &= 82.88 \\
f_{\text{DES MOINES}} &= 94.96 \\
\end{align*}
\]

Transportation costs $c_{ij}$ (from transportation_costs.csv, rows: suppliers, columns: stores):

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

##### Decision Variables

- $y_i \in \{0,1\}$ for $i \in F$: 1 if supplier $i$ is open, 0 otherwise.
- $x_{ij} \geq 0$ for $i \in F$, $j \in S$: quantity supplied from $i$ to $j$.

##### Objective

\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in S$,
   \[
   \sum_{i \in F} x_{ij} \geq d_j
   \]
   where
   \[
   \begin{align*}
   d_{\text{CLARINDA}} &= 2397 \\
   d_{\text{FORT MADISON}} &= 1889 \\
   d_{\text{SIOUX CITY}} &= 2518 \\
   d_{\text{TOLEDO}} &= 3218 \\
   d_{\text{BANCROFT}} &= 1813 \\
   \end{align*}
   \]

2. **Activation constraint:**  
   For all $i \in F$, $j \in S$,
   \[
   x_{ij} \leq M_{ij} y_i
   \]
   where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} = d_j$).

3. **Variable domains:**  
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
   \]

##### Full Numerical Model

\[
\begin{align*}
\min\ & 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}} \\
&+ 694.68\,x_{\text{MOUNT AYR},\,\text{CLARINDA}} + 17.48\,x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + 20.07\,x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + 199.02\,x_{\text{MOUNT AYR},\,\text{TOLEDO}} + 1685.53\,x_{\text{MOUNT AYR},\,\text{BANCROFT}} \\
&+ 15.13\,x_{\text{WAUKEE},\,\text{CLARINDA}} + 1.50\,x_{\text{WAUKEE},\,\text{FORT MADISON}} + 1.43\,x_{\text{WAUKEE},\,\text{SIOUX CITY}} + 27.88\,x_{\text{WAUKEE},\,\text{TOLEDO}} + 90.69\,x_{\text{WAUKEE},\,\text{BANCROFT}} \\
&+ 2.34\,x_{\text{WAVERLY},\,\text{CLARINDA}} + 349.34\,x_{\text{WAVERLY},\,\text{FORT MADISON}} + 246.60\,x_{\text{WAVERLY},\,\text{SIOUX CITY}} + 41.30\,x_{\text{WAVERLY},\,\text{TOLEDO}} + 78.73\,x_{\text{WAVERLY},\,\text{BANCROFT}} \\
&+ 1181.60\,x_{\text{PELLA},\,\text{CLARINDA}} + 1458.53\,x_{\text{PELLA},\,\text{FORT MADISON}} + 1646.36\,x_{\text{PELLA},\,\text{SIOUX CITY}} + 1924.55\,x_{\text{PELLA},\,\text{TOLEDO}} + 38.93\,x_{\text{PELLA},\,\text{BANCROFT}} \\
&+ 1030.80\,x_{\text{DES MOINES},\,\text{CLARINDA}} + 43.48\,x_{\text{DES MOINES},\,\text{FORT MADISON}} + 932.43\,x_{\text{DES MOINES},\,\text{SIOUX CITY}} + 55.39\,x_{\text{DES MOINES},\,\text{TOLEDO}} + 103.84\,x_{\text{DES MOINES},\,\text{BANCROFT}}
\end{align*}
\]

Subject to:

For each store:
\[
\begin{align*}
x_{\text{MOUNT AYR},\,\text{CLARINDA}} + x_{\text{WAUKEE},\,\text{CLARINDA}} + x_{\text{WAVERLY},\,\text{CLARINDA}} + x_{\text{PELLA},\,\text{CLARINDA}} + x_{\text{DES MOINES},\,\text{CLARINDA}} &\geq 2397 \\
x_{\text{MOUNT AYR},\,\text{FORT MADISON}} + x_{\text{WAUKEE},\,\text{FORT MADISON}} + x_{\text{WAVERLY},\,\text{FORT MADISON}} + x_{\text{PELLA},\,\text{FORT MADISON}} + x_{\text{DES MOINES},\,\text{FORT MADISON}} &\geq 1889 \\
x_{\text{MOUNT AYR},\,\text{SIOUX CITY}} + x_{\text{WAUKEE},\,\text{SIOUX CITY}} + x_{\text{WAVERLY},\,\text{SIOUX CITY}} + x_{\text{PELLA},\,\text{SIOUX CITY}} + x_{\text{DES MOINES},\,\text{SIOUX CITY}} &\geq 2518 \\
x_{\text{MOUNT AYR},\,\text{TOLEDO}} + x_{\text{WAUKEE},\,\text{TOLEDO}} + x_{\text{WAVERLY},\,\text{TOLEDO}} + x_{\text{PELLA},\,\text{TOLEDO}} + x_{\text{DES MOINES},\,\text{TOLEDO}} &\geq 3218 \\
x_{\text{MOUNT AYR},\,\text{BANCROFT}} + x_{\text{WAUKEE},\,\text{BANCROFT}} + x_{\text{WAVERLY},\,\text{BANCROFT}} + x_{\text{PELLA},\,\text{BANCROFT}} + x_{\text{DES MOINES},\,\text{BANCROFT}} &\geq 1813 \\
\end{align*}
\]

For all $i \in F$, $j \in S$:
\[
x_{ij} \leq d_j\, y_i
\]

\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
\]