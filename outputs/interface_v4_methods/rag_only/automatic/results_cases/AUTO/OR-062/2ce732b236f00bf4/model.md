Let:
- \( I \) be the set of suppliers (facilities):  
  \( I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\} \)
- \( J \) be the set of stores (customers):  
  \( J = \{\text{Customer\_1 (CLARINDA)}, \text{Customer\_2 (FORT MADISON)}, \text{Customer\_3 (SIOUX CITY)}, \text{Customer\_4 (TOLEDO)}, \text{Customer\_5 (BANCROFT)}\} \)

Parameters:
- Fixed costs for each supplier \( f_i \):

\[
\begin{align*}
f_{\text{MOUNT AYR}} &= 96.58 \\
f_{\text{WAUKEE}} &= 94.06 \\
f_{\text{WAVERLY}} &= 94.37 \\
f_{\text{PELLA}} &= 82.88 \\
f_{\text{DES MOINES}} &= 94.96 \\
\end{align*}
\]

- Transportation cost per unit from supplier \( i \) to store \( j \), \( c_{ij} \):

\[
\begin{array}{l|ccccc}
 & \text{Customer\_1} & \text{Customer\_2} & \text{Customer\_3} & \text{Customer\_4} & \text{Customer\_5} \\
 & (\text{CLARINDA}) & (\text{FORT MADISON}) & (\text{SIOUX CITY}) & (\text{TOLEDO}) & (\text{BANCROFT}) \\
\hline
\text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
\text{WAUKEE} & 15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
\text{WAVERLY} & 2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
\text{PELLA} & 1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
\text{DES MOINES} & 1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{array}
\]

- Demand for each store \( d_j \):

\[
\begin{align*}
d_{\text{Customer\_1 (CLARINDA)}} &= 2397 \\
d_{\text{Customer\_2 (FORT MADISON)}} &= 1889 \\
d_{\text{Customer\_3 (SIOUX CITY)}} &= 2518 \\
d_{\text{Customer\_4 (TOLEDO)}} &= 3218 \\
d_{\text{Customer\_5 (BANCROFT)}} &= 1813 \\
\end{align*}
\]

Decision variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is open, 0 otherwise.
- \( x_{ij} \geq 0 \): quantity supplied from supplier \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{subject to} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J \\
\end{align*}
\]

Where all parameters are as specified above, with all vectors and matrices explicitly given. The objective is to minimize the total cost (fixed supplier opening costs plus transportation costs) while meeting the demand at each store.