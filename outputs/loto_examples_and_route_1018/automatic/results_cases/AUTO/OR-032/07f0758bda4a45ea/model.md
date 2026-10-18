Let $x_i$ be the number of units of product $i$ (where $i$ indexes the following five 'Books' products) to fulfill.

Products (in source order):

- Books_15.15
- Books_30.3
- Books_45.45
- Books_60.6
- Books_75.75

Parameters:

- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial Inventory for product $i$

Data:

\[
\begin{array}{lccc}
\text{Product} & r_i & d_i & s_i \\
\hline
\text{Books\_15.15} & 15.15 & 1980 & 9920.0 \\
\text{Books\_30.3} & 30.3 & 3024 & 20160.0 \\
\text{Books\_45.45} & 45.45 & 4536 & 30000.0 \\
\text{Books\_60.6} & 60.6 & 5601 & 38360.0 \\
\text{Books\_75.75} & 75.75 & 7567 & 51450.0 \\
\end{array}
\]

Model:

Objective:
\[
\max \quad 15.15\,x_{\text{Books\_15.15}} + 30.3\,x_{\text{Books\_30.3}} + 45.45\,x_{\text{Books\_45.45}} + 60.6\,x_{\text{Books\_60.6}} + 75.75\,x_{\text{Books\_75.75}}
\]

Subject to, for each product $i$:

\[
0 \leq x_i \leq \min\{d_i,\, s_i\}
\]
or, equivalently,
\[
x_i \leq d_i \\
x_i \leq s_i \\
x_i \geq 0 \\
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, for each product:

\[
\begin{align*}
0 \leq x_{\text{Books\_15.15}} &\leq 1980 \\
0 \leq x_{\text{Books\_30.3}} &\leq 3024 \\
0 \leq x_{\text{Books\_45.45}} &\leq 4536 \\
0 \leq x_{\text{Books\_60.6}} &\leq 5601 \\
0 \leq x_{\text{Books\_75.75}} &\leq 7567 \\
\end{align*}
\]

and all $x_i$ are integers.

Summary:

Maximize total revenue from fulfilling 'Books' products, subject to inventory and demand limits for each product, with integer, nonnegative fulfillment quantities.