##### Decision Variables

$x_i \geq 0$ (integer): Number of units of product $i$ to fulfill, for each $i$ in the set of Organ products.

Let $i \in \{\text{Organic Fruits},\ \text{Organic Staples},\ \text{Organic Vegetables}\}$.

##### Parameters

- Revenue per unit: $r_i$
- Initial Inventory: $I_i$
- Demand: $d_i$

From the data:

\[
\begin{array}{lccc}
\text{Product } (i) & r_i & I_i & d_i \\
\hline
\text{Organic Fruits} & 60.8 & 5,\!034,\!020 & 678,\!906 \\
\text{Organic Staples} & 918.45 & 5,\!589,\!290 & 749,\!927 \\
\text{Organic Vegetables} & 77.52 & 5,\!202,\!710 & 699,\!808 \\
\end{array}
\]

##### Objective Function

\[
\max\ 60.8\,x_{\text{Organic Fruits}} + 918.45\,x_{\text{Organic Staples}} + 77.52\,x_{\text{Organic Vegetables}}
\]

##### Constraints

1. Inventory: $0 \leq x_i \leq I_i$ for all $i$
2. Demand: $0 \leq x_i \leq d_i$ for all $i$
3. Integrality: $x_i$ integer for all $i$

Or, explicitly:

\[
\begin{align*}
0 \leq\ &x_{\text{Organic Fruits}} \leq \min(5,\!034,\!020,\ 678,\!906) \\
0 \leq\ &x_{\text{Organic Staples}} \leq \min(5,\!589,\!290,\ 749,\!927) \\
0 \leq\ &x_{\text{Organic Vegetables}} \leq \min(5,\!202,\!710,\ 699,\!808) \\
x_i &\in \mathbb{Z}_{\geq 0}
\end{align*}
\]

##### Complete Model

\[
\begin{align*}
\max\quad & 60.8\,x_{\text{Organic Fruits}} + 918.45\,x_{\text{Organic Staples}} + 77.52\,x_{\text{Organic Vegetables}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{Organic Fruits}} \leq 678,\!906 \\
& 0 \leq x_{\text{Organic Staples}} \leq 749,\!927 \\
& 0 \leq x_{\text{Organic Vegetables}} \leq 699,\!808 \\
& x_{\text{Organic Fruits}},\ x_{\text{Organic Staples}},\ x_{\text{Organic Vegetables}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]