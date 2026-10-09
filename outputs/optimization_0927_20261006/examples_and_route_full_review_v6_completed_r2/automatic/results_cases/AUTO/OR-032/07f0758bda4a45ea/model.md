Let $x_i$ be the number of units of product $i$ (where $i$ indexes the following five 'Books' products) to fulfill. All variables are nonnegative integers.

Products and parameters:

\[
\begin{array}{llll}
\text{Product\_Name} & \text{Revenue } (r_i) & \text{Demand } (d_i) & \text{Initial Inventory } (s_i) \\
\hline
\text{Books\_15.15} & 15.15 & 1980 & 9920.0 \\
\text{Books\_30.3} & 30.3 & 3024 & 20160.0 \\
\text{Books\_45.45} & 45.45 & 4536 & 30000.0 \\
\text{Books\_60.6} & 60.6 & 5601 & 38360.0 \\
\text{Books\_75.75} & 75.75 & 7567 & 51450.0 \\
\end{array}
\]

Objective:
\[
\max \quad 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}}
\]

Subject to, for each product $i$:

\[
\begin{align*}
0 \leq x_i \leq \min\{d_i,\, s_i\} \\
x_i \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Explicitly, for each product:

\[
\begin{align*}
0 \leq x_{\text{Books\_15.15}} \leq 1980 \\
0 \leq x_{\text{Books\_30.3}} \leq 3024 \\
0 \leq x_{\text{Books\_45.45}} \leq 4536 \\
0 \leq x_{\text{Books\_60.6}} \leq 5601 \\
0 \leq x_{\text{Books\_75.75}} \leq 7567 \\
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]

Where:
- $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $i$ as above)
- $r_i$ = revenue per unit of product $i$ (from 'Revenue' column)
- $d_i$ = demand for product $i$ (from 'Demand' column)
- $s_i$ = initial inventory for product $i$ (from 'Initial Inventory' column)