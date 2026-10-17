Let $I$ be the set of products classified under ‘Organ’, indexed in source order as follows:

\[
I = \{\text{Organic Fruits},\ \text{Organic Staples},\ \text{Organic Vegetables}\}
\]

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
x_i \geq 0 \quad \text{(continuous or integer, as appropriate)}
\]

Objective:

\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:

\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 && \forall i \in I \\
\end{align*}
\]

Numerical formulation:

\[
\max\ 60.8\,x_{\text{Organic Fruits}} + 918.45\,x_{\text{Organic Staples}} + 77.52\,x_{\text{Organic Vegetables}}
\]

subject to

\[
\begin{align*}
& x_{\text{Organic Fruits}} \leq 678{,}906 \\
& x_{\text{Organic Fruits}} \leq 5{,}034{,}020.0 \\
& x_{\text{Organic Staples}} \leq 749{,}927 \\
& x_{\text{Organic Staples}} \leq 5{,}589{,}290.0 \\
& x_{\text{Organic Vegetables}} \leq 699{,}808 \\
& x_{\text{Organic Vegetables}} \leq 5{,}202{,}710.0 \\
& x_{\text{Organic Fruits}} \geq 0 \\
& x_{\text{Organic Staples}} \geq 0 \\
& x_{\text{Organic Vegetables}} \geq 0 \\
\end{align*}
\]

All coefficients and identifiers are preserved in source order.