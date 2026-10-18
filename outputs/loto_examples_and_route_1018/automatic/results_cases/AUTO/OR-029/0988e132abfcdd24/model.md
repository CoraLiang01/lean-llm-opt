Let $x_i$ denote the number of units of FAUX product $i$ to fulfill, for each product $i$ listed below.

##### Sets and Parameters

Let $I$ be the set of FAUX products:

- FAUX FUR JEWEL SWEATER
- FAUX LEATHER BOMBER JACKET
- FAUX LEATHER BOXY FIT JACKET
- FAUX LEATHER JACKET
- FAUX LEATHER OVERSIZED JACKET LIMITED EDITION
- FAUX LEATHER PUFFER JACKET
- FAUX SHEARLING LINED SUEDE BOOTS
- FAUX SHEARLING PLAID JACKET
- FAUX SUEDE BOMBER JACKET
- FAUX SUEDE JACKET
- FAUX SUEDE OVERSHIRT
- FAUX SUEDE PATCH JACKET

For each product $i \in I$:

- $r_i$ = Revenue per unit (see table)
- $d_i$ = Demand (units)
- $s_i$ = Initial Inventory (units)

| Product Name                                   | $r_i$   | $d_i$  | $s_i$   |
|------------------------------------------------|---------|--------|---------|
| FAUX FUR JEWEL SWEATER                         | 35.9    | 3025   | 20970   |
| FAUX LEATHER BOMBER JACKET                     | 69.9    | 9585   | 71970   |
| FAUX LEATHER BOXY FIT JACKET                   | 99.9    | 4486   | 32730   |
| FAUX LEATHER JACKET                            | 99.9    | 10322  | 71130   |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION  | 159.0   | 4868   | 34910   |
| FAUX LEATHER PUFFER JACKET                     | 69.99   | 8482   | 64010   |
| FAUX SHEARLING LINED SUEDE BOOTS               | 99.9    | 2607   | 20760   |
| FAUX SHEARLING PLAID JACKET                    | 89.9    | 1784   | 12490   |
| FAUX SUEDE BOMBER JACKET                       | 69.9    | 6626   | 50300   |
| FAUX SUEDE JACKET                              | 89.9    | 3256   | 24570   |
| FAUX SUEDE OVERSHIRT                           | 69.9    | 2955   | 24430   |
| FAUX SUEDE PATCH JACKET                        | 89.9    | 910    | 7070    |

##### Decision Variables

For each $i \in I$:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ to fulfill

##### Objective

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

For each $i \in I$:
- Inventory constraint: $x_i \leq s_i$
- Demand constraint:  $x_i \leq d_i$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

##### Complete Model

\[
\begin{align*}
\max \quad & 35.9\, x_1 + 69.9\, x_2 + 99.9\, x_3 + 99.9\, x_4 + 159.0\, x_5 + 69.99\, x_6 \\
          & + 99.9\, x_7 + 89.9\, x_8 + 69.9\, x_9 + 89.9\, x_{10} + 69.9\, x_{11} + 89.9\, x_{12} \\
\text{s.t.} \quad
& x_1 \leq 3025 \\
& x_1 \leq 20970 \\
& x_2 \leq 9585 \\
& x_2 \leq 71970 \\
& x_3 \leq 4486 \\
& x_3 \leq 32730 \\
& x_4 \leq 10322 \\
& x_4 \leq 71130 \\
& x_5 \leq 4868 \\
& x_5 \leq 34910 \\
& x_6 \leq 8482 \\
& x_6 \leq 64010 \\
& x_7 \leq 2607 \\
& x_7 \leq 20760 \\
& x_8 \leq 1784 \\
& x_8 \leq 12490 \\
& x_9 \leq 6626 \\
& x_9 \leq 50300 \\
& x_{10} \leq 3256 \\
& x_{10} \leq 24570 \\
& x_{11} \leq 2955 \\
& x_{11} \leq 24430 \\
& x_{12} \leq 910 \\
& x_{12} \leq 7070 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,12\}
\end{align*}
\]

where the mapping of $x_i$ to product names is as listed above.