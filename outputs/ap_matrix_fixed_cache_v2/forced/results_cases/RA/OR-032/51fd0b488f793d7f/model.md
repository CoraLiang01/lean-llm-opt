Let $x_i$ be the number of units of product $i$ (where $i$ indexes the following five products) to fulfill. All variables are nonnegative integers.

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

Numerical values:

\[
\begin{array}{lccc}
\text{Product\_Name} & r_i & d_i & s_i \\
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
\max \; 15.15\,x_{\text{Books\_15.15}} + 30.3\,x_{\text{Books\_30.3}} + 45.45\,x_{\text{Books\_45.45}} + 60.6\,x_{\text{Books\_60.6}} + 75.75\,x_{\text{Books\_75.75}}
\]

Subject to, for each product $i$:

\[
0 \leq x_i \leq \min\{d_i, s_i\}
\]

That is, explicitly:

\[
\begin{align*}
0 \leq x_{\text{Books\_15.15}} &\leq 1980 \\
0 \leq x_{\text{Books\_30.3}} &\leq 3024 \\
0 \leq x_{\text{Books\_45.45}} &\leq 4536 \\
0 \leq x_{\text{Books\_60.6}} &\leq 5601 \\
0 \leq x_{\text{Books\_75.75}} &\leq 7567 \\
\end{align*}
\]

Integrality:

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

Where $i$ runs over the five products listed above.