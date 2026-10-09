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

| Product Name                                   | $r_i$   | $d_i$  | $s_i$  |
|------------------------------------------------|---------|--------|--------|
| FAUX FUR JEWEL SWEATER                         | 35.9    | 3025   | 20970  |
| FAUX LEATHER BOMBER JACKET                     | 69.9    | 9585   | 71970  |
| FAUX LEATHER BOXY FIT JACKET                   | 99.9    | 4486   | 32730  |
| FAUX LEATHER JACKET                            | 99.9    | 10322  | 71130  |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION  | 159.0   | 4868   | 34910  |
| FAUX LEATHER PUFFER JACKET                     | 69.99   | 8482   | 64010  |
| FAUX SHEARLING LINED SUEDE BOOTS               | 99.9    | 2607   | 20760  |
| FAUX SHEARLING PLAID JACKET                    | 89.9    | 1784   | 12490  |
| FAUX SUEDE BOMBER JACKET                       | 69.9    | 6626   | 50300  |
| FAUX SUEDE JACKET                              | 89.9    | 3256   | 24570  |
| FAUX SUEDE OVERSHIRT                           | 69.9    | 2955   | 24430  |
| FAUX SUEDE PATCH JACKET                        | 89.9    | 910    | 7070   |

##### Decision Variables

For each $i \in I$:

$\quad x_i \in \mathbb{Z}_{\geq 0}$

##### Objective

$\max \sum_{i \in I} r_i x_i$

##### Constraints

For each $i \in I$:

1. Inventory constraint: $x_i \leq s_i$
2. Demand constraint:  $x_i \leq d_i$
3. Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & x_i \leq s_i, \quad \forall i \in I \\
                  & x_i \leq d_i, \quad \forall i \in I \\
                  & x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

where $I$ and all parameters are as listed in the table above.