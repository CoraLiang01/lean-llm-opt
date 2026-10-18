Let $x_i$ be the number of units of product $i$ to be fulfilled, where $i$ indexes the following products:

| Product Name                                   | Revenue | Demand | Initial Inventory |
|------------------------------------------------|---------|--------|------------------|
| FAUX FUR JEWEL SWEATER                         | 35.9    | 3025   | 20970            |
| FAUX LEATHER BOMBER JACKET                     | 69.9    | 9585   | 71970            |
| FAUX LEATHER BOXY FIT JACKET                   | 99.9    | 4486   | 32730            |
| FAUX LEATHER JACKET                            | 99.9    | 10322  | 71130            |
| FAUX LEATHER OVERSIZED JACKET LIMITED EDITION  | 159.0   | 4868   | 34910            |
| FAUX LEATHER PUFFER JACKET                     | 69.99   | 8482   | 64010            |
| FAUX SHEARLING LINED SUEDE BOOTS               | 99.9    | 2607   | 20760            |
| FAUX SHEARLING PLAID JACKET                    | 89.9    | 1784   | 12490            |
| FAUX SUEDE BOMBER JACKET                       | 69.9    | 6626   | 50300            |
| FAUX SUEDE JACKET                              | 89.9    | 3256   | 24570            |
| FAUX SUEDE OVERSHIRT                           | 69.9    | 2955   | 24430            |
| FAUX SUEDE PATCH JACKET                        | 89.9    | 910    | 7070             |

Objective:
$$
\max \left(
35.9\, x_1 +
69.9\, x_2 +
99.9\, x_3 +
99.9\, x_4 +
159.0\, x_5 +
69.99\, x_6 +
99.9\, x_7 +
89.9\, x_8 +
69.9\, x_9 +
89.9\, x_{10} +
69.9\, x_{11} +
89.9\, x_{12}
\right)
$$

Subject to, for each $i = 1, \ldots, 12$ (in the order above):

- Demand constraint:
  $$
  x_i \leq \text{Demand}_i
  $$
- Inventory constraint:
  $$
  x_i \leq \text{Initial Inventory}_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

Where the parameters are as given in the table above for each product $i$.