Let $I$ be the set of FAUX products, indexed by $i$, with the following data:

| $i$ | Product Name | Revenue $r_i$ | Demand $d_i$ | Initial Inventory $s_i$ |
|-----|-----------------------------|--------------|-----------|----------------------|
| 1 | FAUX FUR JEWEL SWEATER | 35.9 | 3025 | 20970 |
| 2 | FAUX LEATHER BOMBER JACKET | 69.9 | 9585 | 71970 |
| 3 | FAUX LEATHER BOXY FIT JACKET | 99.9 | 4486 | 32730 |
| 4 | FAUX LEATHER JACKET | 99.9 | 10322 | 71130 |
| 5 | FAUX LEATHER OVERSIZED JACKET LIMITED EDITION | 159.0 | 4868 | 34910 |
| 6 | FAUX LEATHER PUFFER JACKET | 69.99 | 8482 | 64010 |
| 7 | FAUX SHEARLING LINED SUEDE BOOTS | 99.9 | 2607 | 20760 |
| 8 | FAUX SHEARLING PLAID JACKET | 89.9 | 1784 | 12490 |
| 9 | FAUX SUEDE BOMBER JACKET | 69.9 | 6626 | 50300 |
| 10 | FAUX SUEDE JACKET | 89.9 | 3256 | 24570 |
| 11 | FAUX SUEDE OVERSHIRT | 69.9 | 2955 | 24430 |
| 12 | FAUX SUEDE PATCH JACKET | 89.9 | 910 | 7070 |

Decision variables:
$$
x_i \geq 0 \quad \text{(continuous)},\quad \forall i \in I
$$
where $x_i$ is the number of units of product $i$ fulfilled.

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
\[
\begin{align*}
& x_i \leq d_i, && \forall i \in I \quad \text{(do not exceed demand)} \\
& x_i \leq s_i, && \forall i \in I \quad \text{(do not exceed initial inventory)} \\
& x_i \geq 0, && \forall i \in I \quad \text{(nonnegativity)}
\end{align*}
\]

Where the parameters are:

\[
\begin{array}{lllll}
i & \text{Product Name} & r_i & d_i & s_i \\
1 & \text{FAUX FUR JEWEL SWEATER} & 35.9 & 3025 & 20970 \\
2 & \text{FAUX LEATHER BOMBER JACKET} & 69.9 & 9585 & 71970 \\
3 & \text{FAUX LEATHER BOXY FIT JACKET} & 99.9 & 4486 & 32730 \\
4 & \text{FAUX LEATHER JACKET} & 99.9 & 10322 & 71130 \\
5 & \text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION} & 159.0 & 4868 & 34910 \\
6 & \text{FAUX LEATHER PUFFER JACKET} & 69.99 & 8482 & 64010 \\
7 & \text{FAUX SHEARLING LINED SUEDE BOOTS} & 99.9 & 2607 & 20760 \\
8 & \text{FAUX SHEARLING PLAID JACKET} & 89.9 & 1784 & 12490 \\
9 & \text{FAUX SUEDE BOMBER JACKET} & 69.9 & 6626 & 50300 \\
10 & \text{FAUX SUEDE JACKET} & 89.9 & 3256 & 24570 \\
11 & \text{FAUX SUEDE OVERSHIRT} & 69.9 & 2955 & 24430 \\
12 & \text{FAUX SUEDE PATCH JACKET} & 89.9 & 910 & 7070 \\
\end{array}
\]

All coefficients and identifiers are as retrieved and in source order.