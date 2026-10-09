##### Sets and Indices

Let $I$ be the set of all products classified as ‘FAUX’:
\[
I = \{
\text{FAUX FUR JEWEL SWEATER},
\text{FAUX LEATHER BOMBER JACKET},
\text{FAUX LEATHER BOXY FIT JACKET},
\text{FAUX LEATHER JACKET},
\text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION},
\text{FAUX LEATHER PUFFER JACKET},
\text{FAUX SHEARLING LINED SUEDE BOOTS},
\text{FAUX SHEARLING PLAID JACKET},
\text{FAUX SUEDE BOMBER JACKET},
\text{FAUX SUEDE JACKET},
\text{FAUX SUEDE OVERSHIRT},
\text{FAUX SUEDE PATCH JACKET}
\}
\]

##### Parameters

For each product $i \in I$:

- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial inventory of product $i$

The parameter values are:

| Product Name                                         | $r_i$   | $d_i$  | $s_i$   |
|------------------------------------------------------|---------|--------|---------|
| FAUX FUR JEWEL SWEATER                               | 35.9    | 3025   | 20970   |
| FAUX LEATHER BOMBER JACKET                           | 69.9    | 9585   | 71970   |
| FAUX LEATHER BOXY FIT JACKET                         | 99.9    | 4486   | 32730   |
| FAUX LEATHER JACKET                                  | 99.9    | 10322  | 71130   |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION        | 159.0   | 4868   | 34910   |
| FAUX LEATHER PUFFER JACKET                           | 69.99   | 8482   | 64010   |
| FAUX SHEARLING LINED SUEDE BOOTS                     | 99.9    | 2607   | 20760   |
| FAUX SHEARLING PLAID JACKET                          | 89.9    | 1784   | 12490   |
| FAUX SUEDE BOMBER JACKET                             | 69.9    | 6626   | 50300   |
| FAUX SUEDE JACKET                                    | 89.9    | 3256   | 24570   |
| FAUX SUEDE OVERSHIRT                                 | 69.9    | 2955   | 24430   |
| FAUX SUEDE PATCH JACKET                              | 89.9    | 910    | 7070    |

##### Decision Variables

For each $i \in I$:

- $x_i \geq 0$: Number of units of product $i$ to fulfill (continuous or integer, as appropriate).

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i \in I} r_i x_i \\
\text{s.t.}\quad & 0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I \\
& x_i \geq 0 \qquad \forall i \in I
\end{align*}
\]

##### Parameter Values

\[
\begin{array}{l|ccc}
\text{Product Name} & r_i & d_i & s_i \\
\hline
\text{FAUX FUR JEWEL SWEATER} & 35.9 & 3025 & 20970 \\
\text{FAUX LEATHER BOMBER JACKET} & 69.9 & 9585 & 71970 \\
\text{FAUX LEATHER BOXY FIT JACKET} & 99.9 & 4486 & 32730 \\
\text{FAUX LEATHER JACKET} & 99.9 & 10322 & 71130 \\
\text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION} & 159.0 & 4868 & 34910 \\
\text{FAUX LEATHER PUFFER JACKET} & 69.99 & 8482 & 64010 \\
\text{FAUX SHEARLING LINED SUEDE BOOTS} & 99.9 & 2607 & 20760 \\
\text{FAUX SHEARLING PLAID JACKET} & 89.9 & 1784 & 12490 \\
\text{FAUX SUEDE BOMBER JACKET} & 69.9 & 6626 & 50300 \\
\text{FAUX SUEDE JACKET} & 89.9 & 3256 & 24570 \\
\text{FAUX SUEDE OVERSHIRT} & 69.9 & 2955 & 24430 \\
\text{FAUX SUEDE PATCH JACKET} & 89.9 & 910 & 7070 \\
\end{array}
\]

Where for each $i$, $x_i$ is bounded above by the smaller of demand and initial inventory for that product.