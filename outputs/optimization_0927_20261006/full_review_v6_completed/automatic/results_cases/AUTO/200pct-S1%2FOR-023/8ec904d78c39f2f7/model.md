##### Sets

Let $F$ be the set of suppliers (facilities):
\[
F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}
\]

Let $S$ be the set of stores (customers):
\[
S = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}
\]

##### Parameters

Demands for each store:
\[
\begin{align*}
d_{\text{Customer\_1}} &= 2397 \\
d_{\text{Customer\_2}} &= 1889 \\
d_{\text{Customer\_3}} &= 2518 \\
d_{\text{Customer\_4}} &= 3218 \\
d_{\text{Customer\_5}} &= 1813 \\
\end{align*}
\]

Fixed costs for each supplier:
\[
\begin{align*}
f_{\text{MOUNT AYR}} &= 96.58 \\
f_{\text{WAUKEE}} &= 94.06 \\
f_{\text{WAVERLY}} &= 94.37 \\
f_{\text{PELLA}} &= 82.88 \\
f_{\text{DES MOINES}} &= 94.96 \\
\end{align*}
\]

Transportation costs from each supplier to each store (per unit):

\[
\begin{array}{l|ccccc}
 & \text{CLARINDA} & \text{FORT MADISON} & \text{SIOUX CITY} & \text{TOLEDO} & \text{BANCROFT} \\
\hline
\text{MOUNT AYR} & 694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
\text{WAUKEE} & 15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
\text{WAVERLY} & 2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
\text{PELLA} & 1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
\text{DES MOINES} & 1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{array}
\]

##### Decision Variables

- $x_{fs} \geq 0$: Quantity shipped from supplier $f \in F$ to store $s \in S$ (continuous).
- $y_f \in \{0,1\}$: 1 if supplier $f$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{f \in F} \sum_{s \in S} c_{fs} x_{fs} + \sum_{f \in F} f_f y_f
\]
where $c_{fs}$ is the transportation cost per unit from supplier $f$ to store $s$ as given above.

##### Constraints

1. **Demand satisfaction:** Each store's demand must be fully met:
   \[
   \sum_{f \in F} x_{fs} = d_s, \quad \forall s \in S
   \]

2. **Supplier activation:** No shipments from inactive suppliers:
   \[
   \sum_{s \in S} x_{fs} \leq M y_f, \quad \forall f \in F
   \]
   where $M = \sum_{s \in S} d_s = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$.

3. **Variable domains:**
   \[
   x_{fs} \geq 0 \quad \text{(continuous)}, \quad y_f \in \{0,1\}
   \]

##### Parameters (full listing)

- $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$
- Demands: $d_{\text{Customer\_1}} = 2397$, $d_{\text{Customer\_2}} = 1889$, $d_{\text{Customer\_3}} = 2518$, $d_{\text{Customer\_4}} = 3218$, $d_{\text{Customer\_5}} = 1813$
- Fixed costs: $f_{\text{MOUNT AYR}} = 96.58$, $f_{\text{WAUKEE}} = 94.06$, $f_{\text{WAVERLY}} = 94.37$, $f_{\text{PELLA}} = 82.88$, $f_{\text{DES MOINES}} = 94.96$
- Transportation costs $c_{fs}$ as in the table above.
- $M = 11835$

##### Model Summary

\[
\begin{align*}
\min\ & \sum_{f \in F} \sum_{s \in S} c_{fs} x_{fs} + \sum_{f \in F} f_f y_f \\
\text{s.t.}\quad
& \sum_{f \in F} x_{fs} = d_s, \quad \forall s \in S \\
& \sum_{s \in S} x_{fs} \leq M y_f, \quad \forall f \in F \\
& x_{fs} \geq 0,\quad y_f \in \{0,1\}
\end{align*}
\]

All parameters, sets, and coefficients are as listed above.