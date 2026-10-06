Let $x_i$ be the number of units of product $i$ (as listed below) to fulfill.

Objective:
$$
\max \sum_{i} r_i x_i
$$
where $r_i$ is the Revenue for product $i$.

Subject to, for each product $i$:

- Inventory constraint: $x_i \leq \text{Initial Inventory}_i$
- Demand constraint: $x_i \leq \text{Demand}_i$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

Numerical formulation (in source order):

Let the products be indexed in the order below:

| $i$ | Product Name                                         | $r_i$ (Revenue) | Initial Inventory | Demand |
|-----|------------------------------------------------------|-----------------|------------------|--------|
| 1   | FAUX FUR JEWEL SWEATER                              | 35.9            | 20970            | 3025   |
| 2   | FAUX LEATHER BOMBER JACKET                          | 69.9            | 71970            | 9585   |
| 3   | FAUX LEATHER BOXY FIT JACKET                        | 99.9            | 32730            | 4486   |
| 4   | FAUX LEATHER JACKET                                 | 99.9            | 71130            | 10322  |
| 5   | FAUX LEATHER OVERSIZED JACKET LIMITED EDITION       | 159.0           | 34910            | 4868   |
| 6   | FAUX LEATHER PUFFER JACKET                          | 69.99           | 64010            | 8482   |
| 7   | FAUX SHEARLING LINED SUEDE BOOTS                    | 99.9            | 20760            | 2607   |
| 8   | FAUX SHEARLING PLAID JACKET                         | 89.9            | 12490            | 1784   |
| 9   | FAUX SUEDE BOMBER JACKET                            | 69.9            | 50300            | 6626   |
| 10  | FAUX SUEDE JACKET                                   | 89.9            | 24570            | 3256   |
| 11  | FAUX SUEDE OVERSHIRT                                | 69.9            | 24430            | 2955   |
| 12  | FAUX SUEDE PATCH JACKET                             | 89.9            | 7070             | 910    |

Variables:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,12
$$

Objective:
$$
\max \Big(
35.9\,x_1 + 69.9\,x_2 + 99.9\,x_3 + 99.9\,x_4 + 159.0\,x_5 + 69.99\,x_6 + 99.9\,x_7 + 89.9\,x_8 + 69.9\,x_9 + 89.9\,x_{10} + 69.9\,x_{11} + 89.9\,x_{12}
\Big)
$$

Subject to, for each $i=1,\ldots,12$:
\[
\begin{align*}
x_1 &\leq 20970 \\
x_1 &\leq 3025 \\
x_2 &\leq 71970 \\
x_2 &\leq 9585 \\
x_3 &\leq 32730 \\
x_3 &\leq 4486 \\
x_4 &\leq 71130 \\
x_4 &\leq 10322 \\
x_5 &\leq 34910 \\
x_5 &\leq 4868 \\
x_6 &\leq 64010 \\
x_6 &\leq 8482 \\
x_7 &\leq 20760 \\
x_7 &\leq 2607 \\
x_8 &\leq 12490 \\
x_8 &\leq 1784 \\
x_9 &\leq 50300 \\
x_9 &\leq 6626 \\
x_{10} &\leq 24570 \\
x_{10} &\leq 3256 \\
x_{11} &\leq 24430 \\
x_{11} &\leq 2955 \\
x_{12} &\leq 7070 \\
x_{12} &\leq 910 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,12
\end{align*}
\]

That is, for each product $i$, $x_i$ is the number of units fulfilled, not exceeding both initial inventory and demand, maximizing total revenue.