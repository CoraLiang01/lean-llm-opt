Let $i$ index the Organ product categories:
- $i=1$: Organic Fruits
- $i=2$: Organic Staples
- $i=3$: Organic Vegetables

Let $x_i$ = number of units of Organ product $i$ to fulfill (decision variable, nonnegative integer).

Parameters:
- $r_i$ = Revenue per unit of product $i$
    - $r_1 = 60.8$ (Organic Fruits)
    - $r_2 = 918.45$ (Organic Staples)
    - $r_3 = 77.52$ (Organic Vegetables)
- $d_i$ = Demand for product $i$
    - $d_1 = 678{,}906$
    - $d_2 = 749{,}927$
    - $d_3 = 699{,}808$
- $s_i$ = Initial Inventory for product $i$
    - $s_1 = 5{,}034{,}020.0$
    - $s_2 = 5{,}589{,}290.0$
    - $s_3 = 5{,}202{,}710.0$

Model:

Objective:
\[
\max \; 60.8\,x_1 + 918.45\,x_2 + 77.52\,x_3
\]

Subject to:
\[
\begin{align*}
0 \leq x_1 \leq \min\{678{,}906,\; 5{,}034{,}020.0\} \\
0 \leq x_2 \leq \min\{749{,}927,\; 5{,}589{,}290.0\} \\
0 \leq x_3 \leq \min\{699{,}808,\; 5{,}202{,}710.0\} \\
x_1,\,x_2,\,x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_i$ is the number of units of Organ product $i$ fulfilled, for $i=1,2,3$.
- Each $x_i$ cannot exceed either the demand or the initial inventory for that product, and must be a nonnegative integer.