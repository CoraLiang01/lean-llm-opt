##### Decision Variables

$x_i \geq 0$ (integer): Number of units of each ‘Books’ product $i$ to fulfill, for all $i \in \mathcal{B}$ (the set of ‘Books’ products).

##### Parameters

Let $\mathcal{B} = \{\text{Books\_15.15},\ \text{Books\_30.3},\ \text{Books\_45.45},\ \text{Books\_60.6},\ \text{Books\_75.75}\}$.

For each $i \in \mathcal{B}$:

- $r_i$: Revenue per unit of product $i$.
- $s_i$: Initial inventory of product $i$.
- $d_i$: Demand for product $i$.

Values:

\[
\begin{array}{l|c|c|c}
\text{Product} & r_i\ (\text{Revenue}) & s_i\ (\text{Initial Inventory}) & d_i\ (\text{Demand}) \\
\hline
\text{Books\_15.15} & 15.15 & 9920 & 1980 \\
\text{Books\_30.3} & 30.3 & 20160 & 3024 \\
\text{Books\_45.45} & 45.45 & 30000 & 4536 \\
\text{Books\_60.6} & 60.6 & 38360 & 5601 \\
\text{Books\_75.75} & 75.75 & 51450 & 7567 \\
\end{array}
\]

##### Objective Function

\[
\max \sum_{i \in \mathcal{B}} r_i x_i
\]

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in \mathcal{B}
   \]
   (i.e., for each product, the fulfilled quantity cannot exceed either the available inventory or the demand.)

2. Integer variables:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{B}
   \]

##### Explicit bounds for each product

\[
\begin{array}{l|c}
\text{Product} & 0 \leq x_i \leq \min\{s_i, d_i\} \\
\hline
\text{Books\_15.15} & 0 \leq x_{\text{Books\_15.15}} \leq 1980 \\
\text{Books\_30.3} & 0 \leq x_{\text{Books\_30.3}} \leq 3024 \\
\text{Books\_45.45} & 0 \leq x_{\text{Books\_45.45}} \leq 4536 \\
\text{Books\_60.6} & 0 \leq x_{\text{Books\_60.6}} \leq 5601 \\
\text{Books\_75.75} & 0 \leq x_{\text{Books\_75.75}} \leq 7567 \\
\end{array}
\]

##### Complete Model

\[
\begin{align*}
\max\quad & 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{Books\_15.15}} \leq 1980 \\
& 0 \leq x_{\text{Books\_30.3}} \leq 3024 \\
& 0 \leq x_{\text{Books\_45.45}} \leq 4536 \\
& 0 \leq x_{\text{Books\_60.6}} \leq 5601 \\
& 0 \leq x_{\text{Books\_75.75}} \leq 7567 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in \mathcal{B}
\end{align*}
\]