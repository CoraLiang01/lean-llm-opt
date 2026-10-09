Let $I$ be the set of products classified as ‘FAUX’, indexed by $i$ with names as below.

Let $x_i$ = number of units of product $i$ to be fulfilled (decision variable, nonnegative integer).

Parameters (from data):

| Product Name                                      | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|---------------------------------------------------|-----------------|---------------|--------------------------|
| FAUX FUR JEWEL SWEATER                            | 35.9            | 3025          | 20970                    |
| FAUX LEATHER BOMBER JACKET                        | 69.9            | 9585          | 71970                    |
| FAUX LEATHER BOXY FIT JACKET                      | 99.9            | 4486          | 32730                    |
| FAUX LEATHER JACKET                               | 99.9            | 10322         | 71130                    |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION     | 159.0           | 4868          | 34910                    |
| FAUX LEATHER PUFFER JACKET                        | 69.99           | 8482          | 64010                    |
| FAUX SHEARLING LINED SUEDE BOOTS                  | 99.9            | 2607          | 20760                    |
| FAUX SHEARLING PLAID JACKET                       | 89.9            | 1784          | 12490                    |
| FAUX SUEDE BOMBER JACKET                          | 69.9            | 6626          | 50300                    |
| FAUX SUEDE JACKET                                 | 89.9            | 3256          | 24570                    |
| FAUX SUEDE OVERSHIRT                              | 69.9            | 2955          | 24430                    |
| FAUX SUEDE PATCH JACKET                           | 89.9            | 910           | 7070                     |

Mathematical Model:

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to, for all $i \in I$:
$$
0 \leq x_i \leq \min\{d_i, s_i\}
$$
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

Where:
- $r_i$ = Revenue for product $i$ (see table above)
- $d_i$ = Demand for product $i$ (see table above)
- $s_i$ = Initial Inventory for product $i$ (see table above)
- $x_i$ = Number of units of product $i$ fulfilled

Explicitly, for each product $i$:

- $x_i \leq d_i$
- $x_i \leq s_i$
- $x_i \geq 0$, integer

All variables and parameters use the values as retrieved and shown above.