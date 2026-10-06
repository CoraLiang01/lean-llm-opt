Let $i$ index the following products (in source order):

1. FAUX LEATHER BOMBER JACKET
2. FAUX LEATHER BOXY FIT JACKET
3. FAUX LEATHER JACKET
4. FAUX LEATHER OVERSIZED JACKET LIMITED EDITION
5. FAUX LEATHER PUFFER JACKET
6. FAUX SHEARLING LINED SUEDE BOOTS
7. FAUX SHEARLING PLAID JACKET
8. FAUX SUEDE BOMBER JACKET
9. FAUX SUEDE JACKET
10. FAUX SUEDE OVERSHIRT
11. FAUX SUEDE PATCH JACKET

Let $x_i$ = number of units of product $i$ to fulfill (decision variable, nonnegative integer).

Parameters (from data):

| $i$ | Product Name                                   | Revenue $r_i$ | Demand $d_i$ | Initial Inventory $s_i$ |
|-----|------------------------------------------------|---------------|--------------|------------------------|
| 1   | FAUX LEATHER BOMBER JACKET                     | 69.9          | 9585         | 71970                  |
| 2   | FAUX LEATHER BOXY FIT JACKET                   | 99.9          | 4486         | 32730                  |
| 3   | FAUX LEATHER JACKET                            | 99.9          | 10322        | 71130                  |
| 4   | FAUX LEATHER OVERSIZED JACKET LIMITED EDITION  | 159.0         | 4868         | 34910                  |
| 5   | FAUX LEATHER PUFFER JACKET                     | 69.99         | 8482         | 64010                  |
| 6   | FAUX SHEARLING LINED SUEDE BOOTS               | 99.9          | 2607         | 20760                  |
| 7   | FAUX SHEARLING PLAID JACKET                    | 89.9          | 1784         | 12490                  |
| 8   | FAUX SUEDE BOMBER JACKET                       | 69.9          | 6626         | 50300                  |
| 9   | FAUX SUEDE JACKET                              | 89.9          | 3256         | 24570                  |
| 10  | FAUX SUEDE OVERSHIRT                           | 69.9          | 2955         | 24430                  |
| 11  | FAUX SUEDE PATCH JACKET                        | 89.9          | 910          | 7070                   |

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{11} r_i x_i
$$

Subject to, for all $i = 1, \ldots, 11$:
\[
\begin{align*}
x_i &\leq d_i \\
x_i &\leq s_i \\
x_i &\in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $r_i$ = Revenue for product $i$ (see table above)
- $d_i$ = Demand for product $i$ (see table above)
- $s_i$ = Initial Inventory for product $i$ (see table above)
- $x_i$ = Number of units of product $i$ to fulfill (decision variable, nonnegative integer)