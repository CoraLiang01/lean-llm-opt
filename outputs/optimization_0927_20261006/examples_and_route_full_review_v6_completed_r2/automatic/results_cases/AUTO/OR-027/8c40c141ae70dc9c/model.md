Let $x_i$ denote the number of units of Organ product $i$ to be fulfilled, where $i$ indexes the following products in source order:

- $i=1$: Organic Fruits
- $i=2$: Organic Staples
- $i=3$: Organic Vegetables

Parameters:
- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial Inventory for product $i$

Numerical values (from the data):

\[
\begin{array}{lccc}
\text{Product} & r_i & d_i & s_i \\
\hline
\text{Organic Fruits} & 60.8 & 678{,}906 & 5{,}034{,}020.0 \\
\text{Organic Staples} & 918.45 & 749{,}927 & 5{,}589{,}290.0 \\
\text{Organic Vegetables} & 77.52 & 699{,}808 & 5{,}202{,}710.0 \\
\end{array}
\]

Model:

Objective:
\[
\max \; 60.8\, x_1 + 918.45\, x_2 + 77.52\, x_3
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{678{,}906,\; 5{,}034{,}020.0\} \\
& 0 \leq x_2 \leq \min\{749{,}927,\; 5{,}589{,}290.0\} \\
& 0 \leq x_3 \leq \min\{699{,}808,\; 5{,}202{,}710.0\} \\
& x_1,\, x_2,\, x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Or, equivalently (since inventory always exceeds demand in this data):

\[
\begin{align*}
& 0 \leq x_1 \leq 678{,}906 \\
& 0 \leq x_2 \leq 749{,}927 \\
& 0 \leq x_3 \leq 699{,}808 \\
& x_1,\, x_2,\, x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_i$ = number of units of Organ product $i$ fulfilled (integer, nonnegative, and cannot exceed both demand and initial inventory for that product).