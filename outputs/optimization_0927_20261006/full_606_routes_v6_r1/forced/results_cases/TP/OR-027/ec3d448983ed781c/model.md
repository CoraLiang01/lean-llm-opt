Let $I$ be the set of products classified under ‘Organ’:
$$
I = \{\text{Organic Fruits},\ \text{Organic Staples},\ \text{Organic Vegetables}\}
$$

Let $x_i$ denote the number of units of product $i \in I$ to be fulfilled.

Parameters (from source order):

\[
\begin{array}{l|c|c|c}
\text{Product } i & \text{Revenue } r_i & \text{Demand } d_i & \text{Initial Inventory } s_i \\
\hline
\text{Organic Fruits} & 60.8 & 678{,}906 & 5{,}034{,}020.0 \\
\text{Organic Staples} & 918.45 & 749{,}927 & 5{,}589{,}290.0 \\
\text{Organic Vegetables} & 77.52 & 699{,}808 & 5{,}202{,}710.0 \\
\end{array}
\]

Decision variables:
\[
x_i \geq 0,\quad \text{continuous, for all } i \in I
\]

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(do not fulfill more than demand)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(do not fulfill more than initial inventory)} \\
& x_i \geq 0, && \forall i \in I \quad \text{(nonnegativity)}
\end{align*}
\]

Numerical formulation:

\[
\begin{align*}
\max\quad & 60.8\,x_{\text{Organic Fruits}} + 918.45\,x_{\text{Organic Staples}} + 77.52\,x_{\text{Organic Vegetables}} \\
\text{s.t.}\quad
& x_{\text{Organic Fruits}} \leq 678{,}906 \\
& x_{\text{Organic Fruits}} \leq 5{,}034{,}020.0 \\
& x_{\text{Organic Staples}} \leq 749{,}927 \\
& x_{\text{Organic Staples}} \leq 5{,}589{,}290.0 \\
& x_{\text{Organic Vegetables}} \leq 699{,}808 \\
& x_{\text{Organic Vegetables}} \leq 5{,}202{,}710.0 \\
& x_{\text{Organic Fruits}} \geq 0 \\
& x_{\text{Organic Staples}} \geq 0 \\
& x_{\text{Organic Vegetables}} \geq 0
\end{align*}
\]