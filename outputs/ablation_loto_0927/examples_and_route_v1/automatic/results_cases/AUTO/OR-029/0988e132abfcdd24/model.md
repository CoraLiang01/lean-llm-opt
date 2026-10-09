Let $I$ be the set of products classified as ‘FAUX’:

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

Let $x_i$ be the number of units of product $i$ to fulfill, for each $i \in I$.

Parameters (for each $i \in I$):

- $r_i$: Revenue per unit
- $d_i$: Demand
- $s_i$: Initial Inventory

Data:

| Product Name                                      | $r_i$   | $d_i$  | $s_i$   |
|---------------------------------------------------|---------|--------|---------|
| FAUX FUR JEWEL SWEATER                            | 35.9    | 3025   | 20970   |
| FAUX LEATHER BOMBER JACKET                        | 69.9    | 9585   | 71970   |
| FAUX LEATHER BOXY FIT JACKET                      | 99.9    | 4486   | 32730   |
| FAUX LEATHER JACKET                               | 99.9    | 10322  | 71130   |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION     | 159.0   | 4868   | 34910   |
| FAUX LEATHER PUFFER JACKET                        | 69.99   | 8482   | 64010   |
| FAUX SHEARLING LINED SUEDE BOOTS                  | 99.9    | 2607   | 20760   |
| FAUX SHEARLING PLAID JACKET                       | 89.9    | 1784   | 12490   |
| FAUX SUEDE BOMBER JACKET                          | 69.9    | 6626   | 50300   |
| FAUX SUEDE JACKET                                 | 89.9    | 3256   | 24570   |
| FAUX SUEDE OVERSHIRT                              | 69.9    | 2955   | 24430   |
| FAUX SUEDE PATCH JACKET                           | 89.9    | 910    | 7070    |

Model:

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to, for all $i \in I$:
\[
0 \leq x_i \leq \min\{d_i,\, s_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, for each product $i$:

- $x_i \leq d_i$  (Demand constraint)
- $x_i \leq s_i$  (Inventory constraint)
- $x_i \geq 0$ and integer

Where $x_i$ is the number of units of product $i$ to fulfill.