##### Sets

Let  
$F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$ (suppliers)  
$S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$ (stores)  
$C = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$ (customers, mapped to stores as below)

##### Parameters

Fixed costs (from fixed_cost.csv):  
$\begin{align*}
f_{\text{MOUNT AYR}} &= 96.58 \\
f_{\text{WAUKEE}} &= 94.06 \\
f_{\text{WAVERLY}} &= 94.37 \\
f_{\text{PELLA}} &= 82.88 \\
f_{\text{DES MOINES}} &= 94.96 \\
\end{align*}$

Transportation costs $c_{ij}$ (from transportation_costs.csv):  
$\begin{array}{l|ccccc}
 & \text{CLARINDA} & \text{FORT MADISON} & \text{SIOUX CITY} & \text{TOLEDO} & \text{BANCROFT} \\
\hline
\text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
\text{WAUKEE} & 15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
\text{WAVERLY} & 2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
\text{PELLA} & 1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
\text{DES MOINES} & 1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{array}$

Demand (from demand.csv):  
$\begin{align*}
d_{\text{CLARINDA}} &= 2397 \\
d_{\text{FORT MADISON}} &= 1889 \\
d_{\text{SIOUX CITY}} &= 2518 \\
d_{\text{TOLEDO}} &= 3218 \\
d_{\text{BANCROFT}} &= 1813 \\
\end{align*}$

##### Decision Variables

$y_i \in \{0,1\}$: 1 if supplier $i \in F$ is open, 0 otherwise  
$x_{ij} \geq 0$: quantity shipped from supplier $i \in F$ to store $j \in S$

##### Mathematical Model

Minimize total cost:
$$
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each store:
$$
\sum_{i \in F} x_{ij} \geq d_j \qquad \forall j \in S
$$

2. Linking: Only open suppliers can ship:
$$
x_{ij} \leq M_{ij} y_i \qquad \forall i \in F,\ \forall j \in S
$$
where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} = d_j$).

3. Binary and nonnegativity:
$$
y_i \in \{0,1\} \qquad \forall i \in F \\
x_{ij} \geq 0 \qquad \forall i \in F,\ \forall j \in S
$$

##### Data

- $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- $f_i$ and $c_{ij}$ as above
- $d_j$ as above

##### Complete Numerical Formulation

Minimize
\[
96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
+ \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]
where
\[
\begin{align*}
&c_{\text{MOUNT AYR},\text{CLARINDA}} = 694.68,\quad c_{\text{MOUNT AYR},\text{FORT MADISON}} = 17.48,\quad c_{\text{MOUNT AYR},\text{SIOUX CITY}} = 20.07,\\
&c_{\text{MOUNT AYR},\text{TOLEDO}} = 199.02,\quad c_{\text{MOUNT AYR},\text{BANCROFT}} = 1685.53,\\
&c_{\text{WAUKEE},\text{CLARINDA}} = 15.13,\quad c_{\text{WAUKEE},\text{FORT MADISON}} = 1.5,\quad c_{\text{WAUKEE},\text{SIOUX CITY}} = 1.43,\\
&c_{\text{WAUKEE},\text{TOLEDO}} = 27.88,\quad c_{\text{WAUKEE},\text{BANCROFT}} = 90.69,\\
&c_{\text{WAVERLY},\text{CLARINDA}} = 2.34,\quad c_{\text{WAVERLY},\text{FORT MADISON}} = 349.34,\quad c_{\text{WAVERLY},\text{SIOUX CITY}} = 246.6,\\
&c_{\text{WAVERLY},\text{TOLEDO}} = 41.3,\quad c_{\text{WAVERLY},\text{BANCROFT}} = 78.73,\\
&c_{\text{PELLA},\text{CLARINDA}} = 1181.6,\quad c_{\text{PELLA},\text{FORT MADISON}} = 1458.53,\quad c_{\text{PELLA},\text{SIOUX CITY}} = 1646.36,\\
&c_{\text{PELLA},\text{TOLEDO}} = 1924.55,\quad c_{\text{PELLA},\text{BANCROFT}} = 38.93,\\
&c_{\text{DES MOINES},\text{CLARINDA}} = 1030.8,\quad c_{\text{DES MOINES},\text{FORT MADISON}} = 43.48,\quad c_{\text{DES MOINES},\text{SIOUX CITY}} = 932.43,\\
&c_{\text{DES MOINES},\text{TOLEDO}} = 55.39,\quad c_{\text{DES MOINES},\text{BANCROFT}} = 103.84
\end{align*}
\]

Subject to
\[
\begin{align*}
&x_{\text{MOUNT AYR},j} + x_{\text{WAUKEE},j} + x_{\text{WAVERLY},j} + x_{\text{PELLA},j} + x_{\text{DES MOINES},j} \geq d_j \quad \forall j \in S \\
&x_{ij} \leq d_j\, y_i \quad \forall i \in F,\ \forall j \in S \\
&y_i \in \{0,1\} \quad \forall i \in F \\
&x_{ij} \geq 0 \quad \forall i \in F,\ \forall j \in S
\end{align*}
\]
where
\[
\begin{align*}
d_{\text{CLARINDA}} &= 2397 \\
d_{\text{FORT MADISON}} &= 1889 \\
d_{\text{SIOUX CITY}} &= 2518 \\
d_{\text{TOLEDO}} &= 3218 \\
d_{\text{BANCROFT}} &= 1813
\end{align*}
\]